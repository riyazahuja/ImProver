/-- A base bump function in an inner product space. This construction works in any space with a
norm smooth away from zero but we do not have a typeclass for this. -/
noncomputable def ContDiffBumpBase.ofInnerProductSpace : ContDiffBumpBase E where
  toFun R x := smoothTransition ((R - ‖x‖) / (R - 1))
  mem_Icc _ _ := ⟨smoothTransition.nonneg _, smoothTransition.le_one _⟩
                      /-
                        E : Type u_1
                        inst✝¹ : NormedAddCommGroup E
                        inst✝ : InnerProductSpace Real E
                        x✝¹ : Real
                        x✝ : E
                        ⊢ Eq ((fun R x => (HDiv.hDiv (HSub.hSub R (Norm.norm x)) (HSub.hSub R 1)).smoo …
                      -/
  symmetric _ _ := by simp only [norm_neg]
                      /-
                        🎉 no goals
                      -/
  smooth := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContDiffOn Real (↑Top.top) (Function.uncurry fun R x => (HDiv.hDiv (HSub.hSu …
    -/
    rintro ⟨R, x⟩ ⟨hR : 1 < R, -⟩
    /-
      case mk.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      x : E
      hR : LT.lt 1 R
      ⊢ ContDiffWithinAt Real (↑Top.top) (Function.uncurry fun R x => (HDiv.hDiv (HS …
    -/
    apply ContDiffAt.contDiffWithinAt
    /-
      case mk.intro.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      x : E
      hR : LT.lt 1 R
      ⊢ ContDiffAt Real (↑Top.top) (Function.uncurry fun R x => (HDiv.hDiv (HSub.hSu …
    -/
    rw [← sub_pos] at hR
    /-
      case mk.intro.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      x : E
      hR : LT.lt 0 (HSub.hSub R 1)
      ⊢ ContDiffAt Real (↑Top.top) (Function.uncurry fun R x => (HDiv.hDiv (HSub.hSu …
    -/
    rcases eq_or_ne x 0 with rfl | hx
    · have A : ContinuousAt (fun p : ℝ × E ↦ (p.1 - ‖p.2‖) / (p.1 - 1)) (R, 0) :=
        (continuousAt_fst.sub continuousAt_snd.norm).div
          (continuousAt_fst.sub continuousAt_const) hR.ne'
      have B : ∀ᶠ p in 𝓝 (R, (0 : E)), 1 ≤ (p.1 - ‖p.2‖) / (p.1 - 1) :=
        A.eventually <| le_mem_nhds <| (one_lt_div hR).2 <| sub_lt_sub_left (by simp) _
      refine (contDiffAt_const (c := 1)).congr_of_eventuallyEq <| B.mono fun _ ↦
        smoothTransition.one_of_one_le
      /-
        case mk.intro.h.inr
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        R : Real
        x : E
        hR : LT.lt 0 (HSub.hSub R 1)
        hx : Ne x 0
        ⊢ ContDiffAt Real (↑Top.top) (Function.uncurry fun R x => (HDiv.hDiv (HSub.hSu …
      -/
    · refine smoothTransition.contDiffAt.comp _ (ContDiffAt.div ?_ ?_ hR.ne')
        /-
          case mk.intro.h.inr.refine_1
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : InnerProductSpace Real E
          R : Real
          x : E
          hR : LT.lt 0 (HSub.hSub R 1)
          hx : Ne x 0
          ⊢ ContDiffAt Real (↑Top.top) (fun a => HSub.hSub a.1 (Norm.norm a.2)) { fst := …
        -/
      · exact contDiffAt_fst.sub (contDiffAt_snd.norm ℝ hx)
        /-
          🎉 no goals
        -/
        /-
          case mk.intro.h.inr.refine_2
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : InnerProductSpace Real E
          R : Real
          x : E
          hR : LT.lt 0 (HSub.hSub R 1)
          hx : Ne x 0
          ⊢ ContDiffAt Real (↑Top.top) (fun a => HSub.hSub a.1 1) { fst := R, snd := x }
        -/
      · exact contDiffAt_fst.sub contDiffAt_const
        /-
          🎉 no goals
        -/
  eq_one _ hR _ hx := smoothTransition.one_of_one_le <| (one_le_div <| sub_pos.2 hR).2 <|
    sub_le_sub_left hx _
  support R hR := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      hR : LT.lt 1 R
      ⊢ Eq (Function.support ((fun R x => (HDiv.hDiv (HSub.hSub R (Norm.norm x)) (HS …
    -/
    ext x
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      hR : LT.lt 1 R
      x : E
      ⊢ Iff (Membership.mem (Function.support ((fun R x => (HDiv.hDiv (HSub.hSub R ( …
    -/
    rw [mem_support, Ne, smoothTransition.zero_iff_nonpos, not_le, mem_ball_zero_iff]
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      R : Real
      hR : LT.lt 1 R
      x : E
      ⊢ Iff (LT.lt 0 (HDiv.hDiv (HSub.hSub R (Norm.norm x)) (HSub.hSub R 1))) (LT.lt …
    -/
    simp [div_pos_iff, sq_lt_sq, abs_of_pos (one_pos.trans hR), hR, hR.not_lt]
    /-
      🎉 no goals
    -/


/-- Any inner product space has smooth bump functions. -/
instance (priority := 100) hasContDiffBump_of_innerProductSpace : HasContDiffBump E :=
  ⟨⟨.ofInnerProductSpace E⟩⟩

