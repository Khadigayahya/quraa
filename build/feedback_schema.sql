-- Quraa user feedback — run once in Supabase: Dashboard → SQL Editor → paste → Run.
-- The website uses the public "anon" key, which may only INSERT rows (no reading, editing or deleting).
-- The team reads rows in the dashboard, and build/apply_feedback.py reads them with the service key.

create table if not exists public.feedback (
  id                bigint generated always as identity primary key,
  created_at        timestamptz not null default now(),
  kind              text not null check (kind in ('reciter', 'content')),
  verdict           text not null check (verdict in ('correct', 'wrong', 'unsure')),
  predicted_person  text  check (char_length(predicted_person) <= 80),
  predicted_name    text  check (char_length(predicted_name) <= 120),
  true_person       text  check (char_length(true_person) <= 80),
  true_name         text  check (char_length(true_name) <= 120),
  similarity        real,
  margin            real,
  was_unknown       boolean,
  n_windows         int   check (n_windows between 0 and 200),
  content_verdict   text  check (content_verdict in ('quran', 'lecture', 'mixed', 'unclear')),
  embedding         real[] check (embedding is null or array_length(embedding, 1) = 192),
  window_embeddings real[] check (window_embeddings is null or array_length(window_embeddings, 1) <= 1536),
  duration_s        real,
  file_hash         text  check (file_hash = '' or char_length(file_hash) = 64),
  file_name         text  check (char_length(file_name) <= 200),
  gallery_version   text  check (char_length(gallery_version) <= 20),
  app_version       text  check (char_length(app_version) <= 20),
  status            text not null default 'pending' check (status in ('pending', 'used', 'rejected'))
);

create index if not exists feedback_created_idx on public.feedback (created_at);
create index if not exists feedback_true_person_idx on public.feedback (true_person);

alter table public.feedback enable row level security;

drop policy if exists "website can add feedback" on public.feedback;
create policy "website can add feedback" on public.feedback
  for insert to anon
  with check (status = 'pending');
-- (no select/update/delete policies for anon → the public key cannot read or change anything)

-- Handy views for the team (dashboard only; not exposed to anon)
create or replace view public.feedback_accuracy as
  select date_trunc('day', created_at) as day,
         count(*) filter (where kind = 'reciter')                                  as ratings,
         count(*) filter (where kind = 'reciter' and verdict = 'correct')          as correct,
         round(100.0 * count(*) filter (where kind = 'reciter' and verdict = 'correct')
               / nullif(count(*) filter (where kind = 'reciter' and verdict <> 'unsure'), 0), 1) as accuracy_pct
  from public.feedback group by 1 order by 1 desc;

create or replace view public.feedback_new_names as
  select true_name, count(*) as times, count(distinct file_hash) as files
  from public.feedback
  where kind = 'reciter' and verdict = 'wrong' and true_person is null and true_name is not null
  group by true_name order by times desc;

revoke all on public.feedback_accuracy, public.feedback_new_names from anon;
