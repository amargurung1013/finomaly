# Finomaly - Initial Data Inspection

## Dataset Schema
Schema({'step': Int64, 'type': String, 'amount': Float64, 'nameOrig': String, 'oldbalanceOrg': Float64, 'newbalanceOrig': Float64, 'nameDest': String, 'oldbalanceDest': Float64, 'newbalanceDest': Float64, 'isFraud': Int64, 'isFlaggedFraud': Int64})

## Total Rows
shape: (1, 1)
┌─────────┐
│ len     │
│ ---     │
│ u32     │
╞═════════╡
│ 6362620 │
└─────────┘

## Missing Values
shape: (1, 11)
┌──────┬──────┬────────┬─────────┬───┬─────────┬────────┬────────┬────────┐
│ step ┆ type ┆ amount ┆ nameOri ┆ … ┆ oldbala ┆ newbal ┆ isFrau ┆ isFlag │
│ ---  ┆ ---  ┆ ---    ┆ g       ┆   ┆ nceDest ┆ anceDe ┆ d      ┆ gedFra │
│ u32  ┆ u32  ┆ u32    ┆ ---     ┆   ┆ ---     ┆ st     ┆ ---    ┆ ud     │
│      ┆      ┆        ┆ u32     ┆   ┆ u32     ┆ ---    ┆ u32    ┆ ---    │
│      ┆      ┆        ┆         ┆   ┆         ┆ u32    ┆        ┆ u32    │
╞══════╪══════╪════════╪═════════╪═══╪═════════╪════════╪════════╪════════╡
│ 0    ┆ 0    ┆ 0      ┆ 0       ┆ … ┆ 0       ┆ 0      ┆ 0      ┆ 0      │
└──────┴──────┴────────┴─────────┴───┴─────────┴────────┴────────┴────────┘

## Transaction Distribution
shape: (5, 2)
┌──────────┬─────────┐
│ type     ┆ len     │
│ ---      ┆ ---     │
│ str      ┆ u32     │
╞══════════╪═════════╡
│ CASH_OUT ┆ 2237500 │
│ PAYMENT  ┆ 2151495 │
│ CASH_IN  ┆ 1399284 │
│ TRANSFER ┆ 532909  │
│ DEBIT    ┆ 41432   │
└──────────┴─────────┘

## Fraud Distribution
shape: (2, 2)
┌─────────┬─────────┐
│ isFraud ┆ len     │
│ ---     ┆ ---     │
│ i64     ┆ u32     │
╞═════════╪═════════╡
│ 0       ┆ 6354407 │
│ 1       ┆ 8213    │
└─────────┴─────────┘

## Amount Statistics
shape: (1, 4)
┌───────────────┬──────────┬─────────┬──────────┐
│ mean          ┆ median   ┆ minimum ┆ maximum  │
│ ---           ┆ ---      ┆ ---     ┆ ---      │
│ f64           ┆ f64      ┆ f64     ┆ f64      │
╞═══════════════╪══════════╪═════════╪══════════╡
│ 179861.903549 ┆ 74871.94 ┆ 0.0     ┆ 9.2446e7 │
└───────────────┴──────────┴─────────┴──────────┘
