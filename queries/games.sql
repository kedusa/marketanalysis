select m.item_title, sum(s.gamecount) as rounds, median(s.totalbet/s.gamecount) as median_bet
  from casino.game_sessions s
  join casino.menu_items m on m.itemid = s.menu_itemid
  join casino.users cu on cu.userid = s.userid
  join gamer.skingroupskins sgs on cu.skinid = sgs.skinid
  join gamer.ir_sys_useraccounts ua on ua.userid = s.userid
 where s.opendate >= to_date('01/01/2025', 'dd/mm/yyyy')
   and sgs.groupid = 25
   and s.realmoney = 1
   and s.gamecount > 0
   and ua.internalaccount != 1
 group by m.item_title
 order by sum(s.gamecount) desc