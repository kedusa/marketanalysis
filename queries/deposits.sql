        select p.name as processor, 
               count(*) as deposit_count,
               round(ratio_to_report(count(*)) over()*100, 1) as pct,
               round(sum(e.amount/cr.conversionrate)) as deposit_amount, 
               round(avg(e.amount/cr.conversionrate)) as average_deposit, 
               round(max(e.amount/cr.conversionrate)) as biggest_deposit
          from gamer.ir_sys_exttrans e
          join gamer.ir_sys_exttranstypes tt on tt.externaltransactiontypeid = e.externaltransactiontypeid
          join gamer.ir_sys_exttransstatuses s on s.statusid = e.externaltransactionstatusid
          join gamer.ir_sys_processors p on p.processorid = e.processorid
          join gamer.currency_conv_rates_current cr on cr.currencyid = e.currencyid
          join casino.users cu on cu.userid = e.userid
          join gamer.skingroupskins sgs on sgs.skinid = cu.skinid
          join gamer.ir_sys_useraccounts ua on ua.userid = e.userid
         where tt.typename in ('Sale', 'Manual deposit')
           and s.status = 'Approved'
           and e.transactiondate >= to_date('01/01/2025', 'dd/mm/yyyy')
           and cr.basecurrencyid = 30
           and sgs.groupid = 25
           and ua.internalaccount != 1
        group by p.name
    """
}