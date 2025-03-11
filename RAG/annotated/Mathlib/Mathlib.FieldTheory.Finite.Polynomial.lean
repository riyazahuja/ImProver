/-- A polynomial over the integers is divisible by `n : ℕ`
if and only if it is zero over `ZMod n`. -/
theorem C_dvd_iff_zmod (n : ℕ) (φ : MvPolynomial σ ℤ) :
    C (n : ℤ) ∣ φ ↔ map (Int.castRingHom (ZMod n)) φ = 0 :=
  C_dvd_iff_map_hom_eq_zero _ _ (CharP.intCast_eq_zero_iff (ZMod n) n) _


theorem frobenius_zmod (f : MvPolynomial σ (ZMod p)) : frobenius _ p f = expand p f := by
  /-
    σ : Type u_1
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : MvPolynomial σ (ZMod p)
    ⊢ Eq ((frobenius (MvPolynomial σ (ZMod p)) p) f) ((MvPolynomial.expand p) f)
  -/
  apply induction_on f
    /-
      case h_C
      σ : Type u_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : MvPolynomial σ (ZMod p)
      ⊢ ∀ (a : ZMod p), Eq ((frobenius (MvPolynomial σ (ZMod p)) p) (MvPolynomial.C  …
    -/
  · intro a; rw [expand_C, frobenius_def, ← C_pow, ZMod.pow_card]
             /-
               🎉 no goals
             -/
    /-
      case h_add
      σ : Type u_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : MvPolynomial σ (ZMod p)
      ⊢ ∀ (p_1 q : MvPolynomial σ (ZMod p)), Eq ((frobenius (MvPolynomial σ (ZMod p) …
    -/
  · simp only [map_add]; intro _ _ hf hg; rw [hf, hg]
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case h_X
      σ : Type u_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : MvPolynomial σ (ZMod p)
      ⊢ ∀ (p_1 : MvPolynomial σ (ZMod p)) (n : σ), Eq ((frobenius (MvPolynomial σ (Z …
    -/
  · simp only [expand_X, map_mul]
    /-
      case h_X
      σ : Type u_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : MvPolynomial σ (ZMod p)
      ⊢ ∀ (p_1 : MvPolynomial σ (ZMod p)) (n : σ), Eq ((frobenius (MvPolynomial σ (Z …
    -/
    intro _ _ hf; rw [hf, frobenius_def]
                  /-
                    🎉 no goals
                  -/


theorem expand_zmod (f : MvPolynomial σ (ZMod p)) : expand p f = f ^ p :=
  (frobenius_zmod _).symm


/-- Over a field, this is the indicator function as an `MvPolynomial`. -/
def indicator [CommRing K] (a : σ → K) : MvPolynomial σ K :=
  ∏ n, (1 - (X n - C (a n)) ^ (Fintype.card K - 1))


theorem eval_indicator_apply_eq_one (a : σ → K) : eval a (indicator a) = 1 := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : CommRing K
    a : σ → K
    ⊢ Eq ((MvPolynomial.eval a) (MvPolynomial.indicator a)) 1
  -/
  nontriviality
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : CommRing K
    a : σ → K
    a✝ : Nontrivial K
    ⊢ Eq ((MvPolynomial.eval a) (MvPolynomial.indicator a)) 1
  -/
  have : 0 < Fintype.card K - 1 := tsub_pos_of_lt Fintype.one_lt_card
  simp only [indicator, map_prod, map_sub, map_one, map_pow, eval_X, eval_C, sub_self,
    zero_pow this.ne', sub_zero, Finset.prod_const_one]


theorem degrees_indicator (c : σ → K) :
    degrees (indicator c) ≤ ∑ s : σ, (Fintype.card K - 1) • {s} := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : CommRing K
    c : σ → K
    ⊢ LE.le (MvPolynomial.indicator c).degrees (Finset.univ.sum fun s => HSMul.hSM …
  -/
  rw [indicator]
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : CommRing K
    c : σ → K
    ⊢ LE.le (Finset.univ.prod fun n => HSub.hSub 1 (HPow.hPow (HSub.hSub (MvPolyno …
  -/
  refine le_trans (degrees_prod _ _) (Finset.sum_le_sum fun s _ => ?_)
  classical
  refine le_trans (degrees_sub _ _) ?_
  rw [degrees_one, ← bot_eq_zero, bot_sup_eq]
  refine le_trans (degrees_pow _ _) (nsmul_le_nsmul_right ?_ _)
  refine le_trans (degrees_sub _ _) ?_
  rw [degrees_C, ← bot_eq_zero, sup_bot_eq]
  exact degrees_X' _


theorem indicator_mem_restrictDegree (c : σ → K) :
    indicator c ∈ restrictDegree σ K (Fintype.card K - 1) := by
  classical
  rw [mem_restrictDegree_iff_sup, indicator]
  intro n
  refine le_trans (Multiset.count_le_of_le _ <| degrees_indicator _) (le_of_eq ?_)
  simp_rw [← Multiset.coe_countAddMonoidHom, map_sum,
    AddMonoidHom.map_nsmul, Multiset.coe_countAddMonoidHom, nsmul_eq_mul, Nat.cast_id]
  trans
  · refine Finset.sum_eq_single n ?_ ?_
    · intro b _ ne
      simp [Multiset.count_singleton, ne, if_neg (Ne.symm _)]
    · intro h; exact (h <| Finset.mem_univ _).elim
  · rw [Multiset.count_singleton_self, mul_one]


theorem eval_indicator_apply_eq_zero (a b : σ → K) (h : a ≠ b) : eval a (indicator b) = 0 := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : Field K
    a b : σ → K
    h : Ne a b
    ⊢ Eq ((MvPolynomial.eval a) (MvPolynomial.indicator b)) 0
  -/
  obtain ⟨i, hi⟩ : ∃ i, a i ≠ b i := by rwa [Ne, funext_iff, not_forall] at h
  simp only [indicator, map_prod, map_sub, map_one, map_pow, eval_X, eval_C, sub_self,
    Finset.prod_eq_zero_iff]
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : Field K
    a b : σ → K
    h : Ne a b
    i : σ
    hi : Ne (a i) (b i)
    ⊢ Exists fun a_1 => And (Membership.mem Finset.univ a_1) (Eq (HSub.hSub 1 (HPo …
  -/
  refine ⟨i, Finset.mem_univ _, ?_⟩
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : Field K
    a b : σ → K
    h : Ne a b
    i : σ
    hi : Ne (a i) (b i)
    ⊢ Eq (HSub.hSub 1 (HPow.hPow (HSub.hSub (a i) (b i)) (HSub.hSub (Fintype.card  …
  -/
  rw [FiniteField.pow_card_sub_one_eq_one, sub_self]
  /-
    case intro.ha
    K : Type u_1
    σ : Type u_2
    inst✝² : Fintype K
    inst✝¹ : Fintype σ
    inst✝ : Field K
    a b : σ → K
    h : Ne a b
    i : σ
    hi : Ne (a i) (b i)
    ⊢ Ne (HSub.hSub (a i) (b i)) 0
  -/
  rwa [Ne, sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- `MvPolynomial.eval` as a `K`-linear map. -/
@[simps]
def evalₗ [CommSemiring K] : MvPolynomial σ K →ₗ[K] (σ → K) → K where
  toFun p e := eval e p
                     /-
                       K : Type u_1
                       σ : Type u_2
                       inst✝ : CommSemiring K
                       p q : MvPolynomial σ K
                       ⊢ Eq ((fun p e => (MvPolynomial.eval e) p) (HAdd.hAdd p q)) (HAdd.hAdd ((fun p …
                     -/
  map_add' p q := by ext x; simp
                            /-
                              🎉 no goals
                            -/
                      /-
                        K : Type u_1
                        σ : Type u_2
                        inst✝ : CommSemiring K
                        a : K
                        p : MvPolynomial σ K
                        ⊢ Eq ({ toFun := fun p e => (MvPolynomial.eval e) p, map_add' := ⋯ }.toFun (HS …
                      -/
  map_smul' a p := by ext e; simp
                             /-
                               🎉 no goals
                             -/


theorem map_restrict_dom_evalₗ : (restrictDegree σ K (Fintype.card K - 1)).map (evalₗ K σ) = ⊤ := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝² : Field K
    inst✝¹ : Fintype K
    inst✝ : Finite σ
    ⊢ Eq (Submodule.map (MvPolynomial.evalₗ K σ) (MvPolynomial.restrictDegree σ K  …
  -/
  cases nonempty_fintype σ
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝² : Field K
    inst✝¹ : Fintype K
    inst✝ : Finite σ
    val✝ : Fintype σ
    ⊢ Eq (Submodule.map (MvPolynomial.evalₗ K σ) (MvPolynomial.restrictDegree σ K  …
  -/
  refine top_unique (SetLike.le_def.2 fun e _ => mem_map.2 ?_)
  classical
  refine ⟨∑ n : σ → K, e n • indicator n, ?_, ?_⟩
  · exact sum_mem fun c _ => smul_mem _ _ (indicator_mem_restrictDegree _)
  · ext n
    simp only [_root_.map_sum, @Finset.sum_apply (σ → K) (fun _ => K) _ _ _ _ _, Pi.smul_apply,
      map_smul]
    simp only [evalₗ_apply]
    trans
    · refine Finset.sum_eq_single n (fun b _ h => ?_) ?_
      · rw [eval_indicator_apply_eq_zero _ _ h.symm, smul_zero]
      · exact fun h => (h <| Finset.mem_univ n).elim
    · rw [eval_indicator_apply_eq_one, smul_eq_mul, mul_one]


/-- The submodule of multivariate polynomials whose degree of each variable is strictly less
than the cardinality of K. -/
def R [CommRing K] : Type u :=
  restrictDegree σ K (Fintype.card K - 1)


noncomputable instance [CommRing K] : AddCommGroup (R σ K) :=
  inferInstanceAs (AddCommGroup (restrictDegree σ K (Fintype.card K - 1)))


noncomputable instance [CommRing K] : Module K (R σ K) :=
  inferInstanceAs (Module K (restrictDegree σ K (Fintype.card K - 1)))


noncomputable instance [CommRing K] : Inhabited (R σ K) :=
  inferInstanceAs (Inhabited (restrictDegree σ K (Fintype.card K - 1)))


/-- Evaluation in the `MvPolynomial.R` subtype. -/
def evalᵢ [CommRing K] : R σ K →ₗ[K] (σ → K) → K :=
  (evalₗ K σ).comp (restrictDegree σ K (Fintype.card K - 1)).subtype

-- TODO: would be nice to replace this by suitable decidability assumptions

open Classical in
noncomputable instance decidableRestrictDegree (m : ℕ) :
    DecidablePred (· ∈ { n : σ →₀ ℕ | ∀ i, n i ≤ m }) := by
  /-
    σ K : Type u
    inst✝ : Fintype K
    m : Nat
    ⊢ DecidablePred fun x => Membership.mem (setOf fun n => ∀ (i : σ), LE.le (n i) …
  -/
  simp only [Set.mem_setOf_eq]; infer_instance
                                /-
                                  🎉 no goals
                                -/


open Classical in
theorem rank_R [Fintype σ] : Module.rank K (R σ K) = Fintype.card (σ → K) :=
  calc
    Module.rank K (R σ K) =
        Module.rank K (↥{ s : σ →₀ ℕ | ∀ n : σ, s n ≤ Fintype.card K - 1 } →₀ K) :=
      LinearEquiv.rank_eq
        (Finsupp.supportedEquivFinsupp { s : σ →₀ ℕ | ∀ n : σ, s n ≤ Fintype.card K - 1 })
                                                                  /-
                                                                    σ K : Type u
                                                                    inst✝² : Fintype K
                                                                    inst✝¹ : Field K
                                                                    inst✝ : Fintype σ
                                                                    ⊢ Eq (Module.rank K (Finsupp (↑(setOf fun s => ∀ (n : σ), LE.le (s n) (HSub.hS …
                                                                  -/
    _ = #{ s : σ →₀ ℕ | ∀ n : σ, s n ≤ Fintype.card K - 1 } := by rw [rank_finsupp_self']
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    _ = #{ s : σ → ℕ | ∀ n : σ, s n < Fintype.card K } := by
      /-
        σ K : Type u
        inst✝² : Fintype K
        inst✝¹ : Field K
        inst✝ : Fintype σ
        ⊢ Eq (Cardinal.mk ↑(setOf fun s => ∀ (n : σ), LE.le (s n) (HSub.hSub (Fintype. …
      -/
      refine Quotient.sound ⟨Equiv.subtypeEquiv Finsupp.equivFunOnFinite fun f => ?_⟩
      /-
        σ K : Type u
        inst✝² : Fintype K
        inst✝¹ : Field K
        inst✝ : Fintype σ
        f : Finsupp σ Nat
        ⊢ Iff (Membership.mem (setOf fun s => ∀ (n : σ), LE.le (s n) (HSub.hSub (Finty …
      -/
      refine forall_congr' fun n => le_tsub_iff_right ?_
      /-
        σ K : Type u
        inst✝² : Fintype K
        inst✝¹ : Field K
        inst✝ : Fintype σ
        f : Finsupp σ Nat
        n : σ
        ⊢ LE.le 1 (Fintype.card K)
      -/
      exact Fintype.card_pos_iff.2 ⟨0⟩
      /-
        🎉 no goals
      -/
    _ = #(σ → { n // n < Fintype.card K }) :=
      (@Equiv.subtypePiEquivPi σ (fun _ => ℕ) fun _ n => n < Fintype.card K).cardinal_eq
    _ = #(σ → Fin (Fintype.card K)) :=
      (Equiv.arrowCongr (Equiv.refl σ) Fin.equivSubtype.symm).cardinal_eq
    _ = #(σ → K) := (Equiv.arrowCongr (Equiv.refl σ) (Fintype.equivFin K).symm).cardinal_eq
    _ = Fintype.card (σ → K) := Cardinal.mk_fintype _


instance [Finite σ] : FiniteDimensional K (R σ K) := by
  /-
    σ K : Type u
    inst✝² : Fintype K
    inst✝¹ : Field K
    inst✝ : Finite σ
    ⊢ FiniteDimensional K (MvPolynomial.R σ K)
  -/
  cases nonempty_fintype σ
  classical
  exact
    IsNoetherian.iff_fg.1
      (IsNoetherian.iff_rank_lt_aleph0.mpr <| by
        simpa only [rank_R] using Cardinal.nat_lt_aleph0 (Fintype.card (σ → K)))


open Classical in
theorem finrank_R [Fintype σ] : Module.finrank K (R σ K) = Fintype.card (σ → K) :=
  Module.finrank_eq_of_rank_eq (rank_R σ K)

-- Porting note: was `(evalᵢ σ K).range`.

theorem range_evalᵢ [Finite σ] : range (evalᵢ σ K) = ⊤ := by
  /-
    σ K : Type u
    inst✝² : Fintype K
    inst✝¹ : Field K
    inst✝ : Finite σ
    ⊢ Eq (LinearMap.range (MvPolynomial.evalᵢ σ K)) Top.top
  -/
  rw [evalᵢ, LinearMap.range_comp, range_subtype]
  /-
    σ K : Type u
    inst✝² : Fintype K
    inst✝¹ : Field K
    inst✝ : Finite σ
    ⊢ Eq (Submodule.map (MvPolynomial.evalₗ K σ) (MvPolynomial.restrictDegree σ K  …
  -/
  exact map_restrict_dom_evalₗ K σ
  /-
    🎉 no goals
  -/

-- Porting note: was `(evalᵢ σ K).ker`.

theorem ker_evalₗ [Finite σ] : ker (evalᵢ σ K) = ⊥ := by
  /-
    σ K : Type u
    inst✝² : Fintype K
    inst✝¹ : Field K
    inst✝ : Finite σ
    ⊢ Eq (LinearMap.ker (MvPolynomial.evalᵢ σ K)) Bot.bot
  -/
  cases nonempty_fintype σ
  /-
    case intro
    σ K : Type u
    inst✝² : Fintype K
    inst✝¹ : Field K
    inst✝ : Finite σ
    val✝ : Fintype σ
    ⊢ Eq (LinearMap.ker (MvPolynomial.evalᵢ σ K)) Bot.bot
  -/
  refine (ker_eq_bot_iff_range_eq_top_of_finrank_eq_finrank ?_).mpr (range_evalᵢ σ K)
  classical
  rw [Module.finrank_fintype_fun_eq_card, finrank_R]


theorem eq_zero_of_eval_eq_zero [Finite σ] (p : MvPolynomial σ K) (h : ∀ v : σ → K, eval v p = 0)
    (hp : p ∈ restrictDegree σ K (Fintype.card K - 1)) : p = 0 :=
  let p' : R σ K := ⟨p, hp⟩
  have : p' ∈ ker (evalᵢ σ K) := funext h
                                                   /-
                                                     σ K : Type u
                                                     inst✝² : Fintype K
                                                     inst✝¹ : Field K
                                                     inst✝ : Finite σ
                                                     p : MvPolynomial σ K
                                                     h : ∀ (v : σ → K), Eq ((MvPolynomial.eval v) p) 0
                                                     hp : Membership.mem (MvPolynomial.restrictDegree σ K (HSub.hSub (Fintype.card  …
                                                     p' : MvPolynomial.R σ K := ⟨p, hp⟩
                                                     this : Membership.mem (LinearMap.ker (MvPolynomial.evalᵢ σ K)) p'
                                                     ⊢ Eq p' 0
                                                   -/
  show p'.1 = (0 : R σ K).1 from congr_arg _ <| by rwa [ker_evalₗ, mem_bot] at this
                                                   /-
                                                     🎉 no goals
                                                   -/


