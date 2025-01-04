with users_list as (
    select se.exclusiontime, se2.name, se.userid 
    from gamer.selfexclusions se
    join gamer.self_exclusion_types se2 on se2.id = se.self_exclusion_type_id
    join casino.users cu on cu.userid = se.userid
    join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
    join gamer.skins sk on cu.skinid = sk.skinid
    where se.createddate >= to_date('01/01/2025 00:00:00', 'dd/mm/yyyy hh24:mi:ss')
    and sgs.groupid = 25
    and sk.skin not in ('TestCasino.bet.br', 'OjoTest.bet.br')
    and cu.account_closed = 0
), 
users_conn_list as (
    select cn.base_userid as userid, min(cn.userid) as base_userid
    from gamer.vw_connected_users cn
    where cn.base_userid in (select distinct userid from users_list)
    and cn.userid in (select distinct userid from users_list)
    group by cn.base_userid
)
select ul.exclusiontime, ul.name, count(*), count(distinct ult.base_userid) 
from users_list ul
join users_conn_list ult on ult.userid = ul.userid
group by ul.exclusiontime, ul.name