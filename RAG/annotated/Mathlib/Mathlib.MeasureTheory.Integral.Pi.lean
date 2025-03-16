/-- On a finite product space in `n` variables, for a natural number `n`, a product of integrable
functions depending on each coordinate is integrable. -/
theorem Integrable.fin_nat_prod {n : ℕ} {E : Fin n → Type*}
    [∀ i, MeasureSpace (E i)] [∀ i, SigmaFinite (volume : Measure (E i))]
                                           /-
                                             𝕜 : Type u_1
                                             inst✝² : RCLike 𝕜
                                             n : Nat
                                             E : Fin n → Type u_2
                                             inst✝¹ : (i : Fin n) → MeasureTheory.MeasureSpace (E i)
                                             inst✝ : ∀ (i : Fin n), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.vo …
                                             f : (i : Fin n) → E i → 𝕜
                                             i : Fin n
                                             ⊢ MeasureTheory.Measure (E i)
                                           -/
    {f : (i : Fin n) → E i → 𝕜} (hf : ∀ i, Integrable (f i)) :
                                           /-
                                             🎉 no goals
                                           -/
    /-
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      n : Nat
      E : Fin n → Type u_2
      inst✝¹ : (i : Fin n) → MeasureTheory.MeasureSpace (E i)
      inst✝ : ∀ (i : Fin n), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.vo …
      f : (i : Fin n) → E i → 𝕜
      hf : ∀ (i : Fin n), MeasureTheory.Integrable (f i) MeasureTheory.MeasureSpace. …
      ⊢ MeasureTheory.Measure ((i : Fin n) → E i)
    -/
    Integrable (fun (x : (i : Fin n) → E i) ↦ ∏ i, f i (x i)) := by
    /-
      🎉 no goals
    -/
  induction n with
  | zero => simp only [Finset.univ_eq_empty, Finset.prod_empty, volume_pi,
      integrable_const_iff, one_ne_zero, pi_empty_univ, ENNReal.one_lt_top, or_true]
  | succ n n_ih =>
      have := ((measurePreserving_piFinSuccAbove (fun i => (volume : Measure (E i))) 0).symm)
      rw [volume_pi, ← this.integrable_comp_emb (MeasurableEquiv.measurableEmbedding _)]
      simp_rw [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv,
        Fin.prod_univ_succ, Fin.insertNth_zero]
      simp only [Fin.zero_succAbove, cast_eq, Function.comp_def, Fin.cons_zero, Fin.cons_succ]
      have : Integrable (fun (x : (j : Fin n) → E (Fin.succ j)) ↦ ∏ j, f (Fin.succ j) (x j)) :=
        n_ih (fun i ↦ hf _)
      exact Integrable.prod_mul (hf 0) this


/-- On a finite product space, a product of integrable functions depending on each coordinate is
integrable. Version with dependent target. -/
theorem Integrable.fintype_prod_dep {ι : Type*} [Fintype ι] {E : ι → Type*}
    {f : (i : ι) → E i → 𝕜} [∀ i, MeasureSpace (E i)] [∀ i, SigmaFinite (volume : Measure (E i))]
               /-
                 𝕜 : Type u_1
                 inst✝³ : RCLike 𝕜
                 ι : Type u_2
                 inst✝² : Fintype ι
                 E : ι → Type u_3
                 f : (i : ι) → E i → 𝕜
                 inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
                 inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
                 i : ι
                 ⊢ MeasureTheory.Measure (E i)
               -/
    (hf : ∀ i, Integrable (f i)) :
               /-
                 🎉 no goals
               -/
    /-
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      ι : Type u_2
      inst✝² : Fintype ι
      E : ι → Type u_3
      f : (i : ι) → E i → 𝕜
      inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
      hf : ∀ (i : ι), MeasureTheory.Integrable (f i) MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.Measure ((i : ι) → E i)
    -/
    Integrable (fun (x : (i : ι) → E i) ↦ ∏ i, f i (x i)) := by
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    ι : Type u_2
    inst✝² : Fintype ι
    E : ι → Type u_3
    f : (i : ι) → E i → 𝕜
    inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
    hf : ∀ (i : ι), MeasureTheory.Integrable (f i) MeasureTheory.MeasureSpace.volume
    ⊢ MeasureTheory.Integrable (fun x => Finset.univ.prod fun i => f i (x i)) Meas …
  -/
  let e := (equivFin ι).symm
  simp_rw [← (volume_measurePreserving_piCongrLeft _ e).integrable_comp_emb
    (MeasurableEquiv.measurableEmbedding _),
    ← e.prod_comp, MeasurableEquiv.coe_piCongrLeft, Function.comp_def,
    Equiv.piCongrLeft_apply_apply]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    ι : Type u_2
    inst✝² : Fintype ι
    E : ι → Type u_3
    f : (i : ι) → E i → 𝕜
    inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
    hf : ∀ (i : ι), MeasureTheory.Integrable (f i) MeasureTheory.MeasureSpace.volume
    e : Equiv (Fin (Fintype.card ι)) ι := (Fintype.equivFin ι).symm
    ⊢ MeasureTheory.Integrable (fun x => Finset.univ.prod fun x_1 => f (e x_1) (x  …
  -/
  exact .fin_nat_prod (fun i ↦ hf _)
  /-
    🎉 no goals
  -/


/-- On a finite product space, a product of integrable functions depending on each coordinate is
integrable. -/
theorem Integrable.fintype_prod {ι : Type*} [Fintype ι] {E : Type*}
    {f : ι → E → 𝕜} [MeasureSpace E] [SigmaFinite (volume : Measure E)]
               /-
                 𝕜 : Type u_1
                 inst✝³ : RCLike 𝕜
                 ι : Type u_2
                 inst✝² : Fintype ι
                 E : Type u_3
                 f : ι → E → 𝕜
                 inst✝¹ : MeasureTheory.MeasureSpace E
                 inst✝ : MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
                 i : ι
                 ⊢ MeasureTheory.Measure E
               -/
    (hf : ∀ i, Integrable (f i)) :
               /-
                 🎉 no goals
               -/
    /-
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      ι : Type u_2
      inst✝² : Fintype ι
      E : Type u_3
      f : ι → E → 𝕜
      inst✝¹ : MeasureTheory.MeasureSpace E
      inst✝ : MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
      hf : ∀ (i : ι), MeasureTheory.Integrable (f i) MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.Measure (ι → E)
    -/
    Integrable (fun (x : ι → E) ↦ ∏ i, f i (x i)) :=
    /-
      🎉 no goals
    -/
  Integrable.fintype_prod_dep hf


/-- A version of **Fubini's theorem** in `n` variables, for a natural number `n`. -/
theorem integral_fin_nat_prod_eq_prod {n : ℕ} {E : Fin n → Type*}
    [∀ i, MeasureSpace (E i)] [∀ i, SigmaFinite (volume : Measure (E i))]
    (f : (i : Fin n) → E i → 𝕜) :
    ∫ x : (i : Fin n) → E i, ∏ i, f i (x i) = ∏ i, ∫ x, f i x := by
  induction n with
  | zero =>
      simp only [volume_pi, Finset.univ_eq_empty, Finset.prod_empty, integral_const,
        pi_empty_univ, ENNReal.one_toReal, smul_eq_mul, mul_one, pow_zero, one_smul]
  | succ n n_ih =>
      calc
        _ = ∫ x : E 0 × ((i : Fin n) → E (Fin.succ i)),
            f 0 x.1 * ∏ i : Fin n, f (Fin.succ i) (x.2 i) := by
          rw [volume_pi, ← ((measurePreserving_piFinSuccAbove
            (fun i => (volume : Measure (E i))) 0).symm).integral_comp']
          simp_rw [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv,
            Fin.prod_univ_succ, Fin.insertNth_zero, Equiv.coe_fn_mk, Fin.cons_succ, volume_eq_prod,
            volume_pi, Fin.zero_succAbove, cast_eq, Fin.cons_zero]
        _ = (∫ x, f 0 x) * ∏ i : Fin n, ∫ (x : E (Fin.succ i)), f (Fin.succ i) x := by
          rw [← n_ih, ← integral_prod_mul, volume_eq_prod]
        _ = ∏ i, ∫ x, f i x := by rw [Fin.prod_univ_succ]


/-- A version of **Fubini's theorem** with the variables indexed by a general finite type. -/
theorem integral_fintype_prod_eq_prod (ι : Type*) [Fintype ι] {E : ι → Type*}
    (f : (i : ι) → E i → 𝕜) [∀ i, MeasureSpace (E i)] [∀ i, SigmaFinite (volume : Measure (E i))] :
    ∫ x : (i : ι) → E i, ∏ i, f i (x i) = ∏ i, ∫ x, f i x := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    ι : Type u_2
    inst✝² : Fintype ι
    E : ι → Type u_3
    f : (i : ι) → E i → 𝕜
    inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Finset …
  -/
  let e := (equivFin ι).symm
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    ι : Type u_2
    inst✝² : Fintype ι
    E : ι → Type u_3
    f : (i : ι) → E i → 𝕜
    inst✝¹ : (i : ι) → MeasureTheory.MeasureSpace (E i)
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
    e : Equiv (Fin (Fintype.card ι)) ι := (Fintype.equivFin ι).symm
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Finset …
  -/
  rw [← (volume_measurePreserving_piCongrLeft _ e).integral_comp']
  simp_rw [← e.prod_comp, MeasurableEquiv.coe_piCongrLeft, Equiv.piCongrLeft_apply_apply,
    MeasureTheory.integral_fin_nat_prod_eq_prod]


theorem integral_fintype_prod_eq_pow {E : Type*} (ι : Type*) [Fintype ι] (f : E → 𝕜)
    [MeasureSpace E] [SigmaFinite (volume : Measure E)] :
    ∫ x : ι → E, ∏ i, f (x i) = (∫ x, f x) ^ (card ι) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    ι : Type u_3
    inst✝² : Fintype ι
    f : E → 𝕜
    inst✝¹ : MeasureTheory.MeasureSpace E
    inst✝ : MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Finset …
  -/
  rw [integral_fintype_prod_eq_prod, Finset.prod_const, card]
  /-
    🎉 no goals
  -/


