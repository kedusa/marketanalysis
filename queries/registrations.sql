Select rt.description,
       u.userid,
       u2.username,
       ud.user_parent_id,
       u.opendate,
       ua.account_block_reasons,
       ei.description,
       sk.skin,
       t.accounttypename,
       u.realbalance,
       ua.verifiedplayer,
       ua.verified_bank_details,
       ua.mobile_verification
  from gamer.ir_sys_useraccounts ua
  join gamer.userdetails2 ud on ua.userid = ud.userid
  join casino.users u on u.userid = ud.userid
  join gamer.users2 u2 on u2.userid = ud.userid
  join gamer.skins sk on u.skinid = sk.skinid
  join gamer.registration_type rt on ud.registration_type_id = rt.id
  join gamer.eid_status ei on ua.eid_status = ei.id
  join gamer.accounttypes t on t.accounttypeid = ud.casinoaccounttypeid
 where u.opendate >= to_date('01/01/2025 00:00:00', 'dd/mm/yyyy hh24:mi:ss')
  and sk.skinorigin = 15
  and sk.skin not in ('TestCasino.bet.br', 'OjoTest.bet.br')
  and ua.internalaccount != 1