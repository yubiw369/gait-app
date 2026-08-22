-- =========================================================
-- DERNDUL Production Backend
-- Supabase Auth + PostgreSQL + Row Level Security
-- Run this file once in Supabase SQL Editor.
-- =========================================================

create extension if not exists pgcrypto;

-- ---------------------------------------------------------
-- 1. Profiles
-- ---------------------------------------------------------
create table if not exists public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    display_name text not null,
    role text not null default 'ผู้ใช้งานทั่วไป',
    organization text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------
-- 2. User settings
-- ---------------------------------------------------------
create table if not exists public.user_settings (
    user_id uuid primary key references auth.users(id) on delete cascade,
    default_direction text not null default 'เดินไปทางขวา →',
    show_live_tracking boolean not null default true,
    show_angle_labels boolean not null default true,
    min_landmark_visibility double precision not null default 0.50
        check (
            min_landmark_visibility >= 0.30
            and min_landmark_visibility <= 0.90
        ),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------
-- 3. Issue reports
-- ---------------------------------------------------------
create table if not exists public.issue_reports (
    id uuid primary key default gen_random_uuid(),
    report_code text not null unique,
    user_id uuid not null references auth.users(id) on delete cascade,
    contact_email text not null default '',
    category text not null,
    severity text not null,
    subject text not null
        check (char_length(subject) between 3 and 120),
    details text not null
        check (char_length(details) >= 10),
    app_version text not null,
    status text not null default 'new'
        check (
            status in (
                'new',
                'in_progress',
                'resolved',
                'closed'
            )
        ),
    created_at timestamptz not null default now()
);

create index if not exists issue_reports_user_id_created_at_idx
    on public.issue_reports (user_id, created_at desc);

-- ---------------------------------------------------------
-- 4. updated_at helper
-- ---------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists profiles_set_updated_at
    on public.profiles;

create trigger profiles_set_updated_at
before update on public.profiles
for each row
execute function public.set_updated_at();

drop trigger if exists user_settings_set_updated_at
    on public.user_settings;

create trigger user_settings_set_updated_at
before update on public.user_settings
for each row
execute function public.set_updated_at();

-- ---------------------------------------------------------
-- 5. Create profile + default settings after Auth signup
-- ---------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.profiles (
        id,
        display_name,
        role,
        organization
    )
    values (
        new.id,
        coalesce(
            nullif(
                new.raw_user_meta_data ->> 'display_name',
                ''
            ),
            split_part(
                coalesce(new.email, 'user'),
                '@',
                1
            )
        ),
        coalesce(
            nullif(
                new.raw_user_meta_data ->> 'role',
                ''
            ),
            'ผู้ใช้งานทั่วไป'
        ),
        coalesce(
            new.raw_user_meta_data ->> 'organization',
            ''
        )
    )
    on conflict (id) do nothing;

    insert into public.user_settings (
        user_id
    )
    values (
        new.id
    )
    on conflict (user_id) do nothing;

    return new;
end;
$$;

drop trigger if exists on_auth_user_created
    on auth.users;

create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_user();

-- ---------------------------------------------------------
-- 6. Row Level Security
-- ---------------------------------------------------------
alter table public.profiles
    enable row level security;

alter table public.user_settings
    enable row level security;

alter table public.issue_reports
    enable row level security;

-- Profiles: user sees/creates/updates only own row
drop policy if exists profiles_select_own
    on public.profiles;

create policy profiles_select_own
on public.profiles
for select
to authenticated
using (
    (select auth.uid()) = id
);

drop policy if exists profiles_insert_own
    on public.profiles;

create policy profiles_insert_own
on public.profiles
for insert
to authenticated
with check (
    (select auth.uid()) = id
);

drop policy if exists profiles_update_own
    on public.profiles;

create policy profiles_update_own
on public.profiles
for update
to authenticated
using (
    (select auth.uid()) = id
)
with check (
    (select auth.uid()) = id
);

-- Settings: user sees/creates/updates only own row
drop policy if exists settings_select_own
    on public.user_settings;

create policy settings_select_own
on public.user_settings
for select
to authenticated
using (
    (select auth.uid()) = user_id
);

drop policy if exists settings_insert_own
    on public.user_settings;

create policy settings_insert_own
on public.user_settings
for insert
to authenticated
with check (
    (select auth.uid()) = user_id
);

drop policy if exists settings_update_own
    on public.user_settings;

create policy settings_update_own
on public.user_settings
for update
to authenticated
using (
    (select auth.uid()) = user_id
)
with check (
    (select auth.uid()) = user_id
);

-- Issue reports: authenticated user can create/read own reports.
-- User cannot change report status from the app.
drop policy if exists issue_reports_select_own
    on public.issue_reports;

create policy issue_reports_select_own
on public.issue_reports
for select
to authenticated
using (
    (select auth.uid()) = user_id
);

drop policy if exists issue_reports_insert_own
    on public.issue_reports;

create policy issue_reports_insert_own
on public.issue_reports
for insert
to authenticated
with check (
    (select auth.uid()) = user_id
);

-- ---------------------------------------------------------
-- 7. Least-privilege grants
-- ---------------------------------------------------------
revoke all on table public.profiles
    from anon;

revoke all on table public.user_settings
    from anon;

revoke all on table public.issue_reports
    from anon;

grant select, insert, update
    on table public.profiles
    to authenticated;

grant select, insert, update
    on table public.user_settings
    to authenticated;

grant select, insert
    on table public.issue_reports
    to authenticated;

-- service_role keeps administrative access through Supabase.
