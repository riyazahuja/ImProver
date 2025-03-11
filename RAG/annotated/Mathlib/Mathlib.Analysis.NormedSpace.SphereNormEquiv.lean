/-- The natural homeomorphism between nonzero elements of a normed space `E`
and `Metric.sphere (0 : E) 1 × Set.Ioi (0 : ℝ)`.

The forward map sends `⟨x, hx⟩` to `⟨‖x‖⁻¹ • x, ‖x‖⟩`,
the inverse map sends `(x, r)` to `r • x`.

One may think about it as generalization of polar coordinates to any normed space. -/
@[simps apply_fst_coe apply_snd_coe symm_apply_coe]
noncomputable def homeomorphUnitSphereProd :
    ({0}ᶜ : Set E) ≃ₜ (sphere (0 : E) 1 × Ioi (0 : ℝ)) where
  toFun x := (⟨‖x.1‖⁻¹ • x.1, by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm,
      inv_mul_cancel₀ (norm_ne_zero_iff.2 x.2)]⟩, ⟨‖x.1‖, norm_pos_iff.2 x.2⟩)
  invFun x := ⟨x.2.1 • x.1.1, smul_ne_zero x.2.2.out.ne' (ne_of_mem_sphere x.1.2 one_ne_zero)⟩
                                 /-
                                   E : Type u_1
                                   inst✝¹ : NormedAddCommGroup E
                                   inst✝ : NormedSpace Real E
                                   x : ↑(HasCompl.compl (Singleton.singleton 0))
                                   ⊢ Eq ↑((fun x => ⟨HSMul.hSMul ↑x.2 ↑x.1, ⋯⟩) ((fun x => { fst := ⟨HSMul.hSMul  …
                                 -/
  left_inv x := Subtype.eq <| by simp [smul_inv_smul₀ (norm_ne_zero_iff.2 x.2)]
                                 /-
                                   🎉 no goals
                                 -/
  right_inv
  | (⟨x, hx⟩, ⟨r, hr⟩) => by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx : Membership.mem (Metric.sphere 0 1) x
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      ⊢ Eq ((fun x => { fst := ⟨HSMul.hSMul (Inv.inv (Norm.norm ↑x)) ↑x, ⋯⟩, snd :=  …
    -/
    rw [mem_sphere_zero_iff_norm] at hx
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx✝ : Membership.mem (Metric.sphere 0 1) x
      hx : Eq (Norm.norm x) 1
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      ⊢ Eq ((fun x => { fst := ⟨HSMul.hSMul (Inv.inv (Norm.norm ↑x)) ↑x, ⋯⟩, snd :=  …
    -/
    rw [mem_Ioi] at hr
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx✝ : Membership.mem (Metric.sphere 0 1) x
      hx : Eq (Norm.norm x) 1
      r : Real
      hr✝ : Membership.mem (Set.Ioi 0) r
      hr : LT.lt 0 r
      ⊢ Eq ((fun x => { fst := ⟨HSMul.hSMul (Inv.inv (Norm.norm ↑x)) ↑x, ⋯⟩, snd :=  …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp [hx, norm_smul, hr.le, abs_of_pos hr, hr.ne']
            /-
              🎉 no goals
            -/
  continuous_toFun := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      ⊢ Continuous { toFun := fun x => { fst := ⟨HSMul.hSMul (Inv.inv (Norm.norm ↑x) …
    -/
    refine .prod_mk (.codRestrict (.smul (.inv₀ ?_ ?_) ?_) _) ?_
      /-
        case refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        ⊢ Continuous fun x => Norm.norm ↑x
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        ⊢ ∀ (x : ↑(HasCompl.compl (Singleton.singleton 0))), Ne (Norm.norm ↑x) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        ⊢ Continuous Subtype.val
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        ⊢ Continuous fun x => ⟨Norm.norm ↑x, ⋯⟩
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
                          /-
                            E : Type u_1
                            inst✝¹ : NormedAddCommGroup E
                            inst✝ : NormedSpace Real E
                            ⊢ Continuous { toFun := fun x => { fst := ⟨HSMul.hSMul (Inv.inv (Norm.norm ↑x) …
                          -/
  continuous_invFun := by apply Continuous.subtype_mk (by fun_prop)
                          /-
                            🎉 no goals
                          -/

