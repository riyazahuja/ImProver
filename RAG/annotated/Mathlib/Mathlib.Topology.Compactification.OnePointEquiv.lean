/-- The one-point compactification of a division ring `K` is equivalent to
 the projectivivization `ℙ K (K × K)`. -/
def equivProjectivization :
    OnePoint K ≃ ℙ K (K × K) where
                                     /-
                                       K : Type u_1
                                       inst✝¹ : DivisionRing K
                                       inst✝ : DecidableEq K
                                       p : OnePoint K
                                       ⊢ Ne { fst := 1, snd := 0 } 0
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  toFun p := p.elim (mk K (1, 0) (by simp)) (fun t ↦ mk K (t, 1) (by simp))
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  invFun p := by
    refine Projectivization.lift
      (fun u : {v : K × K // v ≠ 0} ↦ if u.1.2 = 0 then ∞ else ((u.1.2)⁻¹ * u.1.1)) ?_ p
    /-
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      p : Projectivization K (Prod K K)
      ⊢ ∀ (a b : Subtype fun v => Ne v 0) (t : K), Eq (↑a) (HSMul.hSMul t ↑b) → Eq ( …
    -/
    rintro ⟨-, hv⟩ ⟨⟨x, y⟩, hw⟩ t rfl
    /-
      case mk.mk.mk
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      p : Projectivization K (Prod K K)
      x y : K
      hw : Ne { fst := x, snd := y } 0
      t : K
      hv : Ne (HSMul.hSMul t ↑⟨{ fst := x, snd := y }, hw⟩) 0
      ⊢ Eq ((fun u => ite (Eq (↑u).2 0) OnePoint.infty ↑(HMul.hMul (Inv.inv (↑u).2)  …
    -/
    have ht : t ≠ 0 := by rintro rfl; simp at hv
    /-
      case mk.mk.mk
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      p : Projectivization K (Prod K K)
      x y : K
      hw : Ne { fst := x, snd := y } 0
      t : K
      hv : Ne (HSMul.hSMul t ↑⟨{ fst := x, snd := y }, hw⟩) 0
      ht : Ne t 0
      ⊢ Eq ((fun u => ite (Eq (↑u).2 0) OnePoint.infty ↑(HMul.hMul (Inv.inv (↑u).2)  …
    -/
                            /-
                              🎉 no goals
                            -/
    by_cases h₀ : y = 0 <;> simp [h₀, ht, mul_assoc]
                            /-
                              🎉 no goals
                            -/
                   /-
                     K : Type u_1
                     inst✝¹ : DivisionRing K
                     inst✝ : DecidableEq K
                     p : OnePoint K
                     ⊢ Eq ((fun p => Projectivization.lift (fun u => ite (Eq (↑u).2 0) OnePoint.inf …
                   -/
                               /-
                                 🎉 no goals
                               -/
  left_inv p := by cases p <;> simp
                               /-
                                 🎉 no goals
                               -/
  right_inv p := by
    /-
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      p : Projectivization K (Prod K K)
      ⊢ Eq ((fun p => p.elim (Projectivization.mk K { fst := 1, snd := 0 } ⋯) fun t  …
    -/
    induction' p using ind with p hp
    /-
      case h
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      p : Prod K K
      hp : Ne p 0
      ⊢ Eq ((fun p => p.elim (Projectivization.mk K { fst := 1, snd := 0 } ⋯) fun t  …
    -/
    obtain ⟨x, y⟩ := p
    /-
      case h.mk
      K : Type u_1
      inst✝¹ : DivisionRing K
      inst✝ : DecidableEq K
      x y : K
      hp : Ne { fst := x, snd := y } 0
      ⊢ Eq ((fun p => p.elim (Projectivization.mk K { fst := 1, snd := 0 } ⋯) fun t  …
    -/
    by_cases h₀ : y = 0 <;> simp only [mk_eq_mk_iff', h₀, Projectivization.lift_mk, if_true,
      if_false, OnePoint.elim_infty, OnePoint.elim_some, Prod.smul_mk, Prod.mk.injEq, smul_eq_mul,
      mul_zero, and_true]
      /-
        case pos
        K : Type u_1
        inst✝¹ : DivisionRing K
        inst✝ : DecidableEq K
        x y : K
        hp : Ne { fst := x, snd := y } 0
        h₀ : Eq y 0
        ⊢ Exists fun a => Eq (HMul.hMul a x) 1
      -/
    · use x⁻¹
      /-
        case h
        K : Type u_1
        inst✝¹ : DivisionRing K
        inst✝ : DecidableEq K
        x y : K
        hp : Ne { fst := x, snd := y } 0
        h₀ : Eq y 0
        ⊢ Eq (HMul.hMul (Inv.inv x) x) 1
      -/
      simp_all
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝¹ : DivisionRing K
        inst✝ : DecidableEq K
        x y : K
        hp : Ne { fst := x, snd := y } 0
        h₀ : Not (Eq y 0)
        ⊢ Exists fun a => And (Eq (HMul.hMul a x) (HMul.hMul (Inv.inv y) x)) (Eq (HMul …
      -/
    · exact ⟨y⁻¹, rfl, inv_mul_cancel₀ h₀⟩
      /-
        🎉 no goals
      -/


@[simp]
lemma equivProjectivization_apply_infinity :
                                                /-
                                                  K : Type u_1
                                                  inst✝¹ : DivisionRing K
                                                  inst✝ : DecidableEq K
                                                  ⊢ Ne { fst := 1, snd := 0 } 0
                                                -/
    equivProjectivization K ∞ = mk K ⟨1, 0⟩ (by simp) :=
                                                /-
                                                  🎉 no goals
                                                -/
  rfl


@[simp]
lemma equivProjectivization_apply_coe (t : K) :
                                                /-
                                                  K : Type u_1
                                                  inst✝¹ : DivisionRing K
                                                  inst✝ : DecidableEq K
                                                  t : K
                                                  ⊢ Ne { fst := t, snd := 1 } 0
                                                -/
    equivProjectivization K t = mk K ⟨t, 1⟩ (by simp) :=
                                                /-
                                                  🎉 no goals
                                                -/
  rfl


@[simp]
lemma equivProjectivization_symm_apply_mk (x y : K) (h : (x, y) ≠ 0) :
    (equivProjectivization K).symm (mk K ⟨x, y⟩ h) = if y = 0 then ∞ else y⁻¹ * x := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    inst✝ : DecidableEq K
    x y : K
    h : Ne { fst := x, snd := y } 0
    ⊢ Eq ((OnePoint.equivProjectivization K).symm (Projectivization.mk K { fst :=  …
  -/
  simp [equivProjectivization]
  /-
    🎉 no goals
  -/


