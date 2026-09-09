# Generate an independent RClimDex/climdex.pcic reference for climatevar CI.
# RClimDex delegates the precipitation index calculations to climdex.pcic.
# The generated data are deterministic and contain complete daily records.

suppressPackageStartupMessages({
  library(PCICt)
  library(climdex.pcic)
})

dates <- seq(as.Date("1961-01-01"), as.Date("1991-12-31"), by = "day")
idx <- seq_along(dates)
# Deterministic precipitation series with zeros, ordinary wet days and extremes.
prcp <- ((idx * 37) %% 200) / 10
prcp[(idx %% 97) == 0] <- prcp[(idx %% 97) == 0] + 25

pcic_dates <- as.PCICt(as.character(dates), format = "%Y-%m-%d", cal = "gregorian")
ci <- climdexInput.raw(
  prec = prcp,
  prec.dates = pcic_dates,
  base.range = c(1961, 1990),
  max.missing.days = c(annual = 15, monthly = 3)
)

year <- 1991
result <- data.frame(
  year = year,
  rx1day = climdex.rx1day(ci, freq = "annual")[as.character(year)],
  rx5day = climdex.rx5day(ci, freq = "annual", center.mean.on.last.day = TRUE)[as.character(year)],
  r10mm = climdex.r10mm(ci, freq = "annual")[as.character(year)],
  r20mm = climdex.r20mm(ci, freq = "annual")[as.character(year)],
  cdd = climdex.cdd(ci, freq = "annual")[as.character(year)],
  cwd = climdex.cwd(ci, freq = "annual")[as.character(year)],
  sdii = climdex.sdii(ci, freq = "annual")[as.character(year)],
  r95p = climdex.r95ptot(ci)[as.character(year)],
  r99p = climdex.r99ptot(ci)[as.character(year)],
  prcptot = climdex.prcptot(ci)[as.character(year)]
)

write.csv(result, "tests/external/rclimdex_reference.csv", row.names = FALSE)
