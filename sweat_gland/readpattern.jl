# Helper function to read a point pattern for one subject
using PointPatternStatistics
using CSV

function readpattern(subjectid)
    window = [(x=(row.x0, row.x1), y=(row.y0, row.y1)) for row
        in CSV.File("data/meta.csv") if row.subjectid==subjectid][1]
    data = [(row.x, row.y) for row
        in CSV.File("data/glands.csv") if row.subjectid==subjectid]
    pp = PointPattern(data, window)
end
