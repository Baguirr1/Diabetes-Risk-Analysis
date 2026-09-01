-- Lists every table in the public schema and whether RLS is enabled,
-- plus whether it's reachable via Supabase's PostgREST API roles
-- (anon/authenticated) and how many policies it has.
--
-- Run with:
--   psql "$DATABASE_URL" -f check_rls.sql
-- or paste into the Supabase SQL editor.

SELECT
    c.relname                                   AS table_name,
    c.relrowsecurity                            AS rls_enabled,
    c.relforcerowsecurity                       AS rls_forced,
    COALESCE(p.policy_count, 0)                 AS policy_count,
    EXISTS (
        SELECT 1 FROM information_schema.role_table_grants g
        WHERE g.table_schema = 'public'
          AND g.table_name = c.relname
          AND g.grantee IN ('anon', 'authenticated')
    )                                            AS granted_to_api_roles
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN (
    SELECT tablename, COUNT(*) AS policy_count
    FROM pg_policies
    WHERE schemaname = 'public'
    GROUP BY tablename
) p ON p.tablename = c.relname
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
ORDER BY rls_enabled ASC, table_name;
