theorem isLocalHom_of_le_jacobson_bot {R : Type*} [CommRing R] (I : Ideal R)
    (h : I ≤ Ideal.jacobson ⊥) : IsLocalHom (Ideal.Quotient.mk I) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h : LE.le I Bot.bot.jacobson
    ⊢ IsLocalHom (Ideal.Quotient.mk I)
  -/
  constructor
  /-
    case map_nonunit
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h : LE.le I Bot.bot.jacobson
    ⊢ ∀ (a : R), IsUnit ((Ideal.Quotient.mk I) a) → IsUnit a
  -/
  intro a h
  have : IsUnit (Ideal.Quotient.mk (Ideal.jacobson ⊥) a) := by
    rw [isUnit_iff_exists_inv] at *
    obtain ⟨b, hb⟩ := h
    obtain ⟨b, rfl⟩ := Ideal.Quotient.mk_surjective b
    use Ideal.Quotient.mk _ b
    rw [← (Ideal.Quotient.mk _).map_one, ← (Ideal.Quotient.mk _).map_mul, Ideal.Quotient.eq] at hb ⊢
    exact h hb
  /-
    case map_nonunit
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h✝ : LE.le I Bot.bot.jacobson
    a : R
    h : IsUnit ((Ideal.Quotient.mk I) a)
    this : IsUnit ((Ideal.Quotient.mk Bot.bot.jacobson) a)
    ⊢ IsUnit a
  -/
  obtain ⟨⟨x, y, h1, h2⟩, rfl : x = _⟩ := this
  /-
    case map_nonunit.intro.mk
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h✝ : LE.le I Bot.bot.jacobson
    a : R
    h : IsUnit ((Ideal.Quotient.mk I) a)
    y : HasQuotient.Quotient R Bot.bot.jacobson
    h1 : Eq (HMul.hMul ((Ideal.Quotient.mk Bot.bot.jacobson) a) y) 1
    h2 : Eq (HMul.hMul y ((Ideal.Quotient.mk Bot.bot.jacobson) a)) 1
    ⊢ IsUnit a
  -/
  obtain ⟨y, rfl⟩ := Ideal.Quotient.mk_surjective y
  rw [← (Ideal.Quotient.mk _).map_mul, ← (Ideal.Quotient.mk _).map_one, Ideal.Quotient.eq,
    Ideal.mem_jacobson_bot] at h1 h2
  /-
    case map_nonunit.intro.mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h✝ : LE.le I Bot.bot.jacobson
    a : R
    h : IsUnit ((Ideal.Quotient.mk I) a)
    y : R
    h1 : ∀ (y_1 : R), IsUnit (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul a y) 1) y …
    h2 : ∀ (y_1 : R), IsUnit (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul y a) 1) y …
    ⊢ IsUnit a
  -/
  specialize h1 1
  /-
    case map_nonunit.intro.mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h✝ : LE.le I Bot.bot.jacobson
    a : R
    h : IsUnit ((Ideal.Quotient.mk I) a)
    y : R
    h2 : ∀ (y_1 : R), IsUnit (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul y a) 1) y …
    h1 : IsUnit (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul a y) 1) 1) 1)
    ⊢ IsUnit a
  -/
  simp? at h1 says simp only [mul_one, sub_add_cancel, IsUnit.mul_iff] at h1
  /-
    case map_nonunit.intro.mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    h✝ : LE.le I Bot.bot.jacobson
    a : R
    h : IsUnit ((Ideal.Quotient.mk I) a)
    y : R
    h2 : ∀ (y_1 : R), IsUnit (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul y a) 1) y …
    h1 : And (IsUnit a) (IsUnit y)
    ⊢ IsUnit a
  -/
  exact h1.1
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_of_le_jacobson_bot := isLocalHom_of_le_jacobson_bot


/-- A ring `R` is *Henselian* at an ideal `I` if the following condition holds:
for every polynomial `f` over `R`, with a *simple* root `a₀` over the quotient ring `R/I`,
there exists a lift `a : R` of `a₀` that is a root of `f`.

(Here, saying that a root `b` of a polynomial `g` is *simple* means that `g.derivative.eval b` is a
unit. Warning: if `R/I` is not a field then it is not enough to assume that `g` has a factorization
into monic linear factors in which `X - b` shows up only once; for example `1` is not a simple root
of `X^2-1` over `ℤ/4ℤ`.) -/
class HenselianRing (R : Type*) [CommRing R] (I : Ideal R) : Prop where
  jac : I ≤ Ideal.jacobson ⊥
  is_henselian :
    ∀ (f : R[X]) (_ : f.Monic) (a₀ : R) (_ : f.eval a₀ ∈ I)
      (_ : IsUnit (Ideal.Quotient.mk I (f.derivative.eval a₀))), ∃ a : R, f.IsRoot a ∧ a - a₀ ∈ I


/-- A local ring `R` is *Henselian* if the following condition holds:
for every polynomial `f` over `R`, with a *simple* root `a₀` over the residue field,
there exists a lift `a : R` of `a₀` that is a root of `f`.
(Recall that a root `b` of a polynomial `g` is *simple* if it is not a double root, so if
`g.derivative.eval b ≠ 0`.)

In other words, `R` is local Henselian if it is Henselian at the ideal `I`,
in the sense of `HenselianRing`. -/
class HenselianLocalRing (R : Type*) [CommRing R] extends IsLocalRing R : Prop where
  is_henselian :
    ∀ (f : R[X]) (_ : f.Monic) (a₀ : R) (_ : f.eval a₀ ∈ maximalIdeal R)
      (_ : IsUnit (f.derivative.eval a₀)), ∃ a : R, f.IsRoot a ∧ a - a₀ ∈ maximalIdeal R

-- see Note [lower instance priority]

instance (priority := 100) Field.henselian (K : Type*) [Field K] : HenselianLocalRing K where
  is_henselian f _ a₀ h₁ _ := by
    /-
      K : Type u_1
      inst✝ : Field K
      f : Polynomial K
      x✝¹ : f.Monic
      a₀ : K
      h₁ : Membership.mem (IsLocalRing.maximalIdeal K) (Polynomial.eval a₀ f)
      x✝ : IsUnit (Polynomial.eval a₀ (Polynomial.derivative f))
      ⊢ Exists fun a => And (f.IsRoot a) (Membership.mem (IsLocalRing.maximalIdeal K …
    -/
    simp only [(maximalIdeal K).eq_bot_of_prime, Ideal.mem_bot] at h₁ ⊢
    /-
      K : Type u_1
      inst✝ : Field K
      f : Polynomial K
      x✝¹ : f.Monic
      a₀ : K
      x✝ : IsUnit (Polynomial.eval a₀ (Polynomial.derivative f))
      h₁ : Eq (Polynomial.eval a₀ f) 0
      ⊢ Exists fun a => And (f.IsRoot a) (Eq (HSub.hSub a a₀) 0)
    -/
    exact ⟨a₀, h₁, sub_self _⟩
    /-
      🎉 no goals
    -/


theorem HenselianLocalRing.TFAE (R : Type u) [CommRing R] [IsLocalRing R] :
    TFAE
      [HenselianLocalRing R,
        ∀ f : R[X], f.Monic → ∀ a₀ : ResidueField R, aeval a₀ f = 0 →
          aeval a₀ (derivative f) ≠ 0 → ∃ a : R, f.IsRoot a ∧ residue R a = a₀,
        ∀ {K : Type u} [Field K],
          ∀ (φ : R →+* K), Surjective φ → ∀ f : R[X], f.Monic → ∀ a₀ : K,
            f.eval₂ φ a₀ = 0 → f.derivative.eval₂ φ a₀ ≠ 0 → ∃ a : R, f.IsRoot a ∧ φ a = a₀] := by
  tfae_have 3 → 2
  | H => H (residue R) Ideal.Quotient.mk_surjective
  tfae_have 2 → 1
  | H => by
    constructor
    intro f hf a₀ h₁ h₂
    specialize H f hf (residue R a₀)
    have aux := flip mem_nonunits_iff.mp h₂
    simp only [aeval_def, ResidueField.algebraMap_eq, eval₂_at_apply, ←
      Ideal.Quotient.eq_zero_iff_mem, ← IsLocalRing.mem_maximalIdeal] at H h₁ aux
    obtain ⟨a, ha₁, ha₂⟩ := H h₁ aux
    refine ⟨a, ha₁, ?_⟩
    rw [← Ideal.Quotient.eq_zero_iff_mem]
    rwa [← sub_eq_zero, ← RingHom.map_sub] at ha₂
  tfae_have 1 → 3
  | hR, K, _K, φ, hφ, f, hf, a₀, h₁, h₂ => by
    obtain ⟨a₀, rfl⟩ := hφ a₀
    have H := HenselianLocalRing.is_henselian f hf a₀
    simp only [← ker_eq_maximalIdeal φ hφ, eval₂_at_apply, RingHom.mem_ker] at H h₁ h₂
    obtain ⟨a, ha₁, ha₂⟩ := H h₁ (by
      contrapose! h₂
      rwa [← mem_nonunits_iff, ← mem_maximalIdeal, ← ker_eq_maximalIdeal φ hφ,
        RingHom.mem_ker] at h₂)
    refine ⟨a, ha₁, ?_⟩
    rwa [φ.map_sub, sub_eq_zero] at ha₂
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    tfae_3_to_2 : (∀ {K : Type u} [inst : Field K] (φ : RingHom R K), Function.Sur …
    tfae_2_to_1 : (∀ (f : Polynomial R), f.Monic → ∀ (a₀ : IsLocalRing.ResidueFiel …
    tfae_1_to_3 : HenselianLocalRing R → ∀ {K : Type u} [inst : Field K] (φ : Ring …
    ⊢ (List.cons (HenselianLocalRing R) (List.cons (∀ (f : Polynomial R), f.Monic  …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


instance (R : Type*) [CommRing R] [hR : HenselianLocalRing R] :
    HenselianRing R (maximalIdeal R) where
  jac := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      ⊢ LE.le (IsLocalRing.maximalIdeal R) Bot.bot.jacobson
    -/
    rw [Ideal.jacobson, le_sInf_iff]
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      ⊢ ∀ (b : Ideal R), Membership.mem (setOf fun J => And (LE.le Bot.bot J) J.IsMa …
    -/
    rintro I ⟨-, hI⟩
    /-
      case intro
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      I : Ideal R
      hI : I.IsMaximal
      ⊢ LE.le (IsLocalRing.maximalIdeal R) I
    -/
    exact (eq_maximalIdeal hI).ge
    /-
      🎉 no goals
    -/
  is_henselian := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      ⊢ ∀ (f : Polynomial R), f.Monic → ∀ (a₀ : R), Membership.mem (IsLocalRing.maxi …
    -/
    intro f hf a₀ h₁ h₂
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      f : Polynomial R
      hf : f.Monic
      a₀ : R
      h₁ : Membership.mem (IsLocalRing.maximalIdeal R) (Polynomial.eval a₀ f)
      h₂ : IsUnit ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.eval …
      ⊢ Exists fun a => And (f.IsRoot a) (Membership.mem (IsLocalRing.maximalIdeal R …
    -/
    refine HenselianLocalRing.is_henselian f hf a₀ h₁ ?_
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      f : Polynomial R
      hf : f.Monic
      a₀ : R
      h₁ : Membership.mem (IsLocalRing.maximalIdeal R) (Polynomial.eval a₀ f)
      h₂ : IsUnit ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.eval …
      ⊢ IsUnit (Polynomial.eval a₀ (Polynomial.derivative f))
    -/
    contrapose! h₂
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      f : Polynomial R
      hf : f.Monic
      a₀ : R
      h₁ : Membership.mem (IsLocalRing.maximalIdeal R) (Polynomial.eval a₀ f)
      h₂ : Not (IsUnit (Polynomial.eval a₀ (Polynomial.derivative f)))
      ⊢ Not (IsUnit ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.ev …
    -/
    rw [← mem_nonunits_iff, ← IsLocalRing.mem_maximalIdeal, ← Ideal.Quotient.eq_zero_iff_mem] at h₂
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      f : Polynomial R
      hf : f.Monic
      a₀ : R
      h₁ : Membership.mem (IsLocalRing.maximalIdeal R) (Polynomial.eval a₀ f)
      h₂ : Eq ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.eval a₀  …
      ⊢ Not (IsUnit ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.ev …
    -/
    rw [h₂]
    /-
      R : Type u_1
      inst✝ : CommRing R
      hR : HenselianLocalRing R
      f : Polynomial R
      hf : f.Monic
      a₀ : R
      h₁ : Membership.mem (IsLocalRing.maximalIdeal R) (Polynomial.eval a₀ f)
      h₂ : Eq ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal R)) (Polynomial.eval a₀  …
      ⊢ Not (IsUnit 0)
    -/
    exact not_isUnit_zero
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

/-- A ring `R` that is `I`-adically complete is Henselian at `I`. -/
instance (priority := 100) IsAdicComplete.henselianRing (R : Type*) [CommRing R] (I : Ideal R)
    [IsAdicComplete I R] : HenselianRing R I where
  jac := IsAdicComplete.le_jacobson_bot _
  is_henselian := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      I : Ideal R
      inst✝ : IsAdicComplete I R
      ⊢ ∀ (f : Polynomial R), f.Monic → ∀ (a₀ : R), Membership.mem I (Polynomial.eva …
    -/
    intro f _ a₀ h₁ h₂
    classical
      let f' := derivative f
      -- we define a sequence `c n` by starting at `a₀` and then continually
      -- applying the function sending `b` to `b - f(b)/f'(b)` (Newton's method).
      -- Note that `f'.eval b` is a unit, because `b` has the same residue as `a₀` modulo `I`.
      let c : ℕ → R := fun n => Nat.recOn n a₀ fun _ b => b - f.eval b * Ring.inverse (f'.eval b)
      have hc : ∀ n, c (n + 1) = c n - f.eval (c n) * Ring.inverse (f'.eval (c n)) := by
        intro n
        simp only [c, Nat.rec_add_one]
      -- we now spend some time determining properties of the sequence `c : ℕ → R`
      -- `hc_mod`: for every `n`, we have `c n ≡ a₀ [SMOD I]`
      -- `hf'c`  : for every `n`, `f'.eval (c n)` is a unit
      -- `hfcI`  : for every `n`, `f.eval (c n)` is contained in `I ^ (n+1)`
      have hc_mod : ∀ n, c n ≡ a₀ [SMOD I] := by
        intro n
        induction' n with n ih
        · rfl
        rw [hc, sub_eq_add_neg, ← add_zero a₀]
        refine ih.add ?_
        rw [SModEq.zero, Ideal.neg_mem_iff]
        refine I.mul_mem_right _ ?_
        rw [← SModEq.zero] at h₁ ⊢
        exact (ih.eval f).trans h₁
      have hf'c : ∀ n, IsUnit (f'.eval (c n)) := by
        intro n
        haveI := isLocalHom_of_le_jacobson_bot I (IsAdicComplete.le_jacobson_bot I)
        apply IsUnit.of_map (Ideal.Quotient.mk I)
        convert h₂ using 1
        exact SModEq.def.mp ((hc_mod n).eval _)
      have hfcI : ∀ n, f.eval (c n) ∈ I ^ (n + 1) := by
        intro n
        induction' n with n ih
        · simpa only [Nat.rec_zero, zero_add, pow_one] using h₁
        rw [← taylor_eval_sub (c n), hc, sub_eq_add_neg, sub_eq_add_neg,
          add_neg_cancel_comm]
        rw [eval_eq_sum, sum_over_range' _ _ _ (lt_add_of_pos_right _ zero_lt_two), ←
          Finset.sum_range_add_sum_Ico _ (Nat.le_add_left _ _)]
        swap
        · intro i
          rw [zero_mul]
        refine Ideal.add_mem _ ?_ ?_
        · erw [Finset.sum_range_succ]
          rw [Finset.range_one, Finset.sum_singleton,
            taylor_coeff_zero, taylor_coeff_one, pow_zero, pow_one, mul_one, mul_neg,
            mul_left_comm, Ring.mul_inverse_cancel _ (hf'c n), mul_one, add_neg_cancel]
          exact Ideal.zero_mem _
        · refine Submodule.sum_mem _ ?_
          simp only [Finset.mem_Ico]
          rintro i ⟨h2i, _⟩
          have aux : n + 2 ≤ i * (n + 1) := by trans 2 * (n + 1) <;> nlinarith only [h2i]
          refine Ideal.mul_mem_left _ _ (Ideal.pow_le_pow_right aux ?_)
          rw [pow_mul']
          exact Ideal.pow_mem_pow ((Ideal.neg_mem_iff _).2 <| Ideal.mul_mem_right _ _ ih) _
      -- we are now in the position to show that `c : ℕ → R` is a Cauchy sequence
      have aux : ∀ m n, m ≤ n → c m ≡ c n [SMOD (I ^ m • ⊤ : Ideal R)] := by
        intro m n hmn
        rw [← Ideal.one_eq_top, Ideal.smul_eq_mul, mul_one]
        obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
        clear hmn
        induction' k with k ih
        · rw [add_zero]
        rw [← add_assoc, hc, ← add_zero (c m), sub_eq_add_neg]
        refine ih.add ?_
        symm
        rw [SModEq.zero, Ideal.neg_mem_iff]
        refine Ideal.mul_mem_right _ _ (Ideal.pow_le_pow_right ?_ (hfcI _))
        rw [add_assoc]
        exact le_self_add
      -- hence the sequence converges to some limit point `a`, which is the `a` we are looking for
      obtain ⟨a, ha⟩ := IsPrecomplete.prec' c (aux _ _)
      refine ⟨a, ?_, ?_⟩
      · show f.IsRoot a
        suffices ∀ n, f.eval a ≡ 0 [SMOD (I ^ n • ⊤ : Ideal R)] by exact IsHausdorff.haus' _ this
        intro n
        specialize ha n
        rw [← Ideal.one_eq_top, Ideal.smul_eq_mul, mul_one] at ha ⊢
        refine (ha.symm.eval f).trans ?_
        rw [SModEq.zero]
        exact Ideal.pow_le_pow_right le_self_add (hfcI _)
      · show a - a₀ ∈ I
        specialize ha (0 + 1)
        rw [hc, pow_one, ← Ideal.one_eq_top, Ideal.smul_eq_mul, mul_one, sub_eq_add_neg] at ha
        rw [← SModEq.sub_mem, ← add_zero a₀]
        refine ha.symm.trans (SModEq.rfl.add ?_)
        rw [SModEq.zero, Ideal.neg_mem_iff]
        exact Ideal.mul_mem_right _ _ h₁

