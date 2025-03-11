instance {X : Type*} [MetricSpace X] [IsUltrametricDist X] : TotallySeparatedSpace X :=
  totallySeparatedSpace_iff_exists_isClopen.mpr fun x y h ↦ by
    /-
      X : Type u_1
      inst✝¹ : MetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      h : Ne x y
      ⊢ Exists fun U => And (IsClopen U) (And (Membership.mem U x) (Membership.mem ( …
    -/
    obtain ⟨r, hr, hr'⟩ := exists_between (dist_pos.mpr h)
    /-
      case intro.intro
      X : Type u_1
      inst✝¹ : MetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      h : Ne x y
      r : Real
      hr : LT.lt 0 r
      hr' : LT.lt r (Dist.dist x y)
      ⊢ Exists fun U => And (IsClopen U) (And (Membership.mem U x) (Membership.mem ( …
    -/
    refine ⟨_, IsUltrametricDist.isClopen_ball x r, ?_, ?_⟩
      /-
        case intro.intro.refine_1
        X : Type u_1
        inst✝¹ : MetricSpace X
        inst✝ : IsUltrametricDist X
        x y : X
        h : Ne x y
        r : Real
        hr : LT.lt 0 r
        hr' : LT.lt r (Dist.dist x y)
        ⊢ Membership.mem (Metric.ball x r) x
      -/
    · simp only [mem_ball, dist_self, hr]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2
        X : Type u_1
        inst✝¹ : MetricSpace X
        inst✝ : IsUltrametricDist X
        x y : X
        h : Ne x y
        r : Real
        hr : LT.lt 0 r
        hr' : LT.lt r (Dist.dist x y)
        ⊢ Membership.mem (HasCompl.compl (Metric.ball x r)) y
      -/
    · simp only [Set.mem_compl, mem_ball, dist_comm, not_lt, hr'.le]
      /-
        🎉 no goals
      -/


