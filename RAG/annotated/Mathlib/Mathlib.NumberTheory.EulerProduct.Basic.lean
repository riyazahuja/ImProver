/-- If `f` is multiplicative and summable, then its values at natural numbers `> 1`
have norm strictly less than `1`. -/
lemma Summable.norm_lt_one {F : Type*} [NormedField F] [CompleteSpace F] {f : ℕ →* F}
    (hsum : Summable f) {p : ℕ} (hp : 1 < p) :
    ‖f p‖ < 1 := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    p : Nat
    hp : LT.lt 1 p
    ⊢ LT.lt (Norm.norm (f p)) 1
  -/
  refine summable_geometric_iff_norm_lt_one.mp ?_
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    p : Nat
    hp : LT.lt 1 p
    ⊢ Summable fun n => HPow.hPow (f p) n
  -/
  simp_rw [← map_pow]
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    p : Nat
    hp : LT.lt 1 p
    ⊢ Summable fun n => f (HPow.hPow p n)
  -/
  exact hsum.comp_injective <| Nat.pow_right_injective hp
  /-
    🎉 no goals
  -/


@[local instance] private lemma instT0Space : T0Space R := MetricSpace.instT0Space


include hf₁ hmul in
/-- We relate a finite product over primes in `s` to an infinite sum over `s`-factored numbers. -/
lemma summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum
    (hsum : ∀ {p : ℕ}, p.Prime → Summable (fun n : ℕ ↦ ‖f (p ^ n)‖)) (s : Finset ℕ) :
    Summable (fun m : factoredNumbers s ↦ ‖f m‖) ∧
      HasSum (fun m : factoredNumbers s ↦ f m)
        (∏ p ∈ s with p.Prime, ∑' n : ℕ, f (p ^ n)) := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
    s : Finset Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
  -/
  induction' s using Finset.induction with p s hp ih
    /-
      case empty
      R : Type u_1
      inst✝¹ : NormedCommRing R
      f : Nat → R
      inst✝ : CompleteSpace R
      hf₁ : Eq (f 1) 1
      hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
      hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
      ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
    -/
  · rw [factoredNumbers_empty]
    /-
      case empty
      R : Type u_1
      inst✝¹ : NormedCommRing R
      f : Nat → R
      inst✝ : CompleteSpace R
      hf₁ : Eq (f 1) 1
      hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
      hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
      ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
    -/
    simp only [not_mem_empty, IsEmpty.forall_iff, forall_const, filter_true_of_mem, prod_empty]
    /-
      case empty
      R : Type u_1
      inst✝¹ : NormedCommRing R
      f : Nat → R
      inst✝ : CompleteSpace R
      hf₁ : Eq (f 1) 1
      hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
      hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
      ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) 1)
    -/
    exact ⟨(Set.finite_singleton 1).summable (‖f ·‖), hf₁ ▸ hasSum_singleton 1 f⟩
    /-
      🎉 no goals
    -/
    /-
      case insert
      R : Type u_1
      inst✝¹ : NormedCommRing R
      f : Nat → R
      inst✝ : CompleteSpace R
      hf₁ : Eq (f 1) 1
      hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
      hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
      p : Nat
      s : Finset Nat
      hp : Not (Membership.mem s p)
      ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
      ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
    -/
  · rw [filter_insert]
    /-
      case insert
      R : Type u_1
      inst✝¹ : NormedCommRing R
      f : Nat → R
      inst✝ : CompleteSpace R
      hf₁ : Eq (f 1) 1
      hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
      hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
      p : Nat
      s : Finset Nat
      hp : Not (Membership.mem s p)
      ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
      ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((ite (Nat. …
    -/
    split_ifs with hpp
      /-
        case pos
        R : Type u_1
        inst✝¹ : NormedCommRing R
        f : Nat → R
        inst✝ : CompleteSpace R
        hf₁ : Eq (f 1) 1
        hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
        hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
        p : Nat
        s : Finset Nat
        hp : Not (Membership.mem s p)
        ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
        hpp : Nat.Prime p
        ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Insert.in …
      -/
    · constructor
      · simp only [← (equivProdNatFactoredNumbers hpp hp).summable_iff, Function.comp_def,
          equivProdNatFactoredNumbers_apply', factoredNumbers.map_prime_pow_mul hmul hpp hp]
        /-
          case pos.left
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          ⊢ Summable fun x => Norm.norm (HMul.hMul (f (HPow.hPow p x.1)) (f ↑x.2))
        -/
        refine Summable.of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun _ ↦ norm_mul_le ..) ?_
        /-
          case pos.left
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          ⊢ Summable fun x => HMul.hMul (Norm.norm (f (HPow.hPow p x.1))) (Norm.norm (f  …
        -/
                                                         /-
                                                           🎉 no goals
                                                         -/
        apply Summable.mul_of_nonneg (hsum hpp) ih.1 <;> exact fun n ↦ norm_nonneg _
                                                         /-
                                                           🎉 no goals
                                                         -/
        /-
          case pos.right
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          ⊢ HasSum (fun m => f ↑m) ((Insert.insert p (Finset.filter (fun p => Nat.Prime  …
        -/
      · have hp' : p ∉ {p ∈ s | p.Prime} := mt (mem_of_mem_filter p) hp
        /-
          case pos.right
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          hp' : Not (Membership.mem (Finset.filter (fun p => Nat.Prime p) s) p)
          ⊢ HasSum (fun m => f ↑m) ((Insert.insert p (Finset.filter (fun p => Nat.Prime  …
        -/
        rw [prod_insert hp', ← (equivProdNatFactoredNumbers hpp hp).hasSum_iff, Function.comp_def]
        conv =>
          enter [1, x]
          rw [equivProdNatFactoredNumbers_apply', factoredNumbers.map_prime_pow_mul hmul hpp hp]
        /-
          case pos.right
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          hp' : Not (Membership.mem (Finset.filter (fun p => Nat.Prime p) s) p)
          ⊢ HasSum (fun x => HMul.hMul (f (HPow.hPow p x.1)) (f ↑x.2)) (HMul.hMul (tsum  …
        -/
        have : T3Space R := instT3Space -- speeds up the following
        /-
          case pos.right
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          hp' : Not (Membership.mem (Finset.filter (fun p => Nat.Prime p) s) p)
          this : T3Space R
          ⊢ HasSum (fun x => HMul.hMul (f (HPow.hPow p x.1)) (f ↑x.2)) (HMul.hMul (tsum  …
        -/
        apply (hsum hpp).of_norm.hasSum.mul ih.2
        -- `exact summable_mul_of_summable_norm (hsum hpp) ih.1` gives a time-out
        /-
          case pos.right
          R : Type u_1
          inst✝¹ : NormedCommRing R
          f : Nat → R
          inst✝ : CompleteSpace R
          hf₁ : Eq (f 1) 1
          hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
          hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
          p : Nat
          s : Finset Nat
          hp : Not (Membership.mem s p)
          ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
          hpp : Nat.Prime p
          hp' : Not (Membership.mem (Finset.filter (fun p => Nat.Prime p) s) p)
          this : T3Space R
          ⊢ Summable fun x => HMul.hMul (f (HPow.hPow p x.1)) (f ↑x.2)
        -/
        apply summable_mul_of_summable_norm (hsum hpp) ih.1
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝¹ : NormedCommRing R
        f : Nat → R
        inst✝ : CompleteSpace R
        hf₁ : Eq (f 1) 1
        hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
        hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
        p : Nat
        s : Finset Nat
        hp : Not (Membership.mem s p)
        ih : And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset …
        hpp : Not (Nat.Prime p)
        ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
      -/
    · rwa [factoredNumbers_insert s hpp]
      /-
        🎉 no goals
      -/


include hf₁ hmul in
/-- A version of `EulerProduct.summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum`
in terms of the value of the series. -/
lemma prod_filter_prime_tsum_eq_tsum_factoredNumbers (hsum : Summable (‖f ·‖)) (s : Finset ℕ) :
    ∏ p ∈ s with p.Prime, ∑' n : ℕ, f (p ^ n) = ∑' m : factoredNumbers s, f m :=
  (summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum hf₁ hmul
    (fun hp ↦ hsum.comp_injective <| Nat.pow_right_injective hp.one_lt) _).2.tsum_eq.symm


/-- The following statement says that summing over `s`-factored numbers such that
`s` contains `primesBelow N` for large enough `N` gets us arbitrarily close to the sum
over all natural numbers (assuming `f` is summable and `f 0 = 0`; the latter since
`0` is not `s`-factored). -/
lemma norm_tsum_factoredNumbers_sub_tsum_lt (hsum : Summable f) (hf₀ : f 0 = 0) {ε : ℝ}
    (εpos : 0 < ε) :
    ∃ N : ℕ, ∀ s : Finset ℕ, primesBelow N ≤ s →
      ‖(∑' m : ℕ, f m) - ∑' m : factoredNumbers s, f m‖ < ε := by
  obtain ⟨N, hN⟩ :=
    summable_iff_nat_tsum_vanishing.mp hsum (Metric.ball 0 ε) <| Metric.ball_mem_nhds 0 εpos
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    N : Nat
    hN : ∀ (t : Set Nat), HasSubset.Subset t (setOf fun n => LE.le N n) → Membersh …
    ⊢ Exists fun N => ∀ (s : Finset Nat), LE.le N.primesBelow s → LT.lt (Norm.norm …
  -/
  simp_rw [mem_ball_zero_iff] at hN
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    N : Nat
    hN : ∀ (t : Set Nat), HasSubset.Subset t (setOf fun n => LE.le N n) → LT.lt (N …
    ⊢ Exists fun N => ∀ (s : Finset Nat), LE.le N.primesBelow s → LT.lt (Norm.norm …
  -/
  refine ⟨N, fun s hs ↦ ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    N : Nat
    hN : ∀ (t : Set Nat), HasSubset.Subset t (setOf fun n => LE.le N n) → LT.lt (N …
    s : Finset Nat
    hs : LE.le N.primesBelow s
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun m => f m) (tsum fun m => f ↑m))) ε
  -/
  have := hN _ <| factoredNumbers_compl hs
  rwa [← tsum_subtype_add_tsum_subtype_compl hsum (factoredNumbers s),
    add_sub_cancel_left, tsum_eq_tsum_diff_singleton (factoredNumbers s)ᶜ hf₀]

-- Versions of the three lemmas above for `smoothNumbers N`


include hf₁ hmul in
/-- We relate a finite product over primes to an infinite sum over smooth numbers. -/
lemma summable_and_hasSum_smoothNumbers_prod_primesBelow_tsum
    (hsum : ∀ {p : ℕ}, p.Prime → Summable (fun n : ℕ ↦ ‖f (p ^ n)‖)) (N : ℕ) :
    Summable (fun m : N.smoothNumbers ↦ ‖f m‖) ∧
      HasSum (fun m : N.smoothNumbers ↦ f m) (∏ p ∈ N.primesBelow, ∑' n : ℕ, f (p ^ n)) := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
    N : Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) (N.primesBe …
  -/
  rw [smoothNumbers_eq_factoredNumbers, primesBelow]
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p  …
    N : Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
  -/
  exact summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum hf₁ hmul hsum _
  /-
    🎉 no goals
  -/


include hf₁ hmul in
/-- A version of `EulerProduct.summable_and_hasSum_smoothNumbers_prod_primesBelow_tsum`
in terms of the value of the series. -/
lemma prod_primesBelow_tsum_eq_tsum_smoothNumbers (hsum : Summable (‖f ·‖)) (N : ℕ) :
    ∏ p ∈ N.primesBelow, ∑' n : ℕ, f (p ^ n) = ∑' m : N.smoothNumbers, f m :=
  (summable_and_hasSum_smoothNumbers_prod_primesBelow_tsum hf₁ hmul
    (fun hp ↦ hsum.comp_injective <| Nat.pow_right_injective hp.one_lt) _).2.tsum_eq.symm


/-- The following statement says that summing over `N`-smooth numbers
for large enough `N` gets us arbitrarily close to the sum over all natural numbers
(assuming `f` is norm-summable and `f 0 = 0`; the latter since `0` is not smooth). -/
lemma norm_tsum_smoothNumbers_sub_tsum_lt (hsum : Summable f) (hf₀ : f 0 = 0)
    {ε : ℝ} (εpos : 0 < ε) :
    ∃ N₀ : ℕ, ∀ N ≥ N₀, ‖(∑' m : ℕ, f m) - ∑' m : N.smoothNumbers, f m‖ < ε := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Exists fun N₀ => ∀ (N : Nat), GE.ge N N₀ → LT.lt (Norm.norm (HSub.hSub (tsum …
  -/
  conv => enter [1, N₀, N]; rw [smoothNumbers_eq_factoredNumbers]
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Exists fun N₀ => ∀ (N : Nat), GE.ge N N₀ → LT.lt (Norm.norm (HSub.hSub (tsum …
  -/
  obtain ⟨N₀, hN₀⟩ := norm_tsum_factoredNumbers_sub_tsum_lt hsum hf₀ εpos
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    N₀ : Nat
    hN₀ : ∀ (s : Finset Nat), LE.le N₀.primesBelow s → LT.lt (Norm.norm (HSub.hSub …
    ⊢ Exists fun N₀ => ∀ (N : Nat), GE.ge N N₀ → LT.lt (Norm.norm (HSub.hSub (tsum …
  -/
  refine ⟨N₀, fun N hN ↦ hN₀ (range N) fun p hp ↦ ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hsum : Summable f
    hf₀ : Eq (f 0) 0
    ε : Real
    εpos : LT.lt 0 ε
    N₀ : Nat
    hN₀ : ∀ (s : Finset Nat), LE.le N₀.primesBelow s → LT.lt (Norm.norm (HSub.hSub …
    N : Nat
    hN : GE.ge N N₀
    p : Nat
    hp : Membership.mem N₀.primesBelow p
    ⊢ Membership.mem (Finset.range N) p
  -/
  exact mem_range.mpr <| (lt_of_mem_primesBelow hp).trans_le hN
  /-
    🎉 no goals
  -/



include hf₁ hmul in
/-- The *Euler Product* for multiplicative (on coprime arguments) functions.

If `f : ℕ → R`, where `R` is a complete normed commutative ring, `f 0 = 0`, `f 1 = 1`, `f` is
multiplicative on coprime arguments, and `‖f ·‖` is summable, then
`∏' p : Nat.Primes, ∑' e, f (p ^ e) = ∑' n, f n`. This version is stated using `HasProd`. -/
theorem eulerProduct_hasProd (hsum : Summable (‖f ·‖)) (hf₀ : f 0 = 0) :
    HasProd (fun p : Primes ↦ ∑' e, f (p ^ e)) (∑' n, f n) := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    ⊢ HasProd (fun p => tsum fun e => f (HPow.hPow (↑p) e)) (tsum fun n => f n)
  -/
  let F : ℕ → R := fun n ↦ ∑' e, f (n ^ e)
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    F : Nat → R := fun n => tsum fun e => f (HPow.hPow n e)
    ⊢ HasProd (fun p => tsum fun e => f (HPow.hPow (↑p) e)) (tsum fun n => f n)
  -/
  change HasProd (F ∘ Subtype.val) _
  rw [hasProd_subtype_iff_mulIndicator,
    show Set.mulIndicator (fun p : ℕ ↦ Irreducible p) =  {p | Nat.Prime p}.mulIndicator from rfl,
    HasProd, Metric.tendsto_atTop]
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    F : Nat → R := fun n => tsum fun e => f (HPow.hPow n e)
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Finset Nat), GE.ge n N → LT …
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    F : Nat → R := fun n => tsum fun e => f (HPow.hPow n e)
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Finset Nat), GE.ge n N → LT.lt (Dist.dist (n.prod fun …
  -/
  obtain ⟨N₀, hN₀⟩ := norm_tsum_factoredNumbers_sub_tsum_lt hsum.of_norm hf₀ hε
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    F : Nat → R := fun n => tsum fun e => f (HPow.hPow n e)
    ε : Real
    hε : GT.gt ε 0
    N₀ : Nat
    hN₀ : ∀ (s : Finset Nat), LE.le N₀.primesBelow s → LT.lt (Norm.norm (HSub.hSub …
    ⊢ Exists fun N => ∀ (n : Finset Nat), GE.ge n N → LT.lt (Dist.dist (n.prod fun …
  -/
  refine ⟨range N₀, fun s hs ↦ ?_⟩
  have : ∏ p ∈ s, {p | Nat.Prime p}.mulIndicator F p = ∏ p ∈ s with p.Prime, F p :=
    prod_mulIndicator_eq_prod_filter s (fun _ ↦ F) _ id
  rw [this, dist_eq_norm, prod_filter_prime_tsum_eq_tsum_factoredNumbers hf₁ hmul hsum,
    norm_sub_rev]
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    F : Nat → R := fun n => tsum fun e => f (HPow.hPow n e)
    ε : Real
    hε : GT.gt ε 0
    N₀ : Nat
    hN₀ : ∀ (s : Finset Nat), LE.le N₀.primesBelow s → LT.lt (Norm.norm (HSub.hSub …
    s : Finset Nat
    hs : GE.ge s (Finset.range N₀)
    this : Eq (s.prod fun p => (setOf fun p => Nat.Prime p).mulIndicator F p) ((Fi …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => f n) (tsum fun m => f ↑m))) ε
  -/
  exact hN₀ s fun p hp ↦ hs <| mem_range.mpr <| lt_of_mem_primesBelow hp
  /-
    🎉 no goals
  -/


include hf₁ hmul in
/-- The *Euler Product* for multiplicative (on coprime arguments) functions.

If `f : ℕ → R`, where `R` is a complete normed commutative ring, `f 0 = 0`, `f 1 = 1`, `f` i
multiplicative on coprime arguments, and `‖f ·‖` is summable, then
`∏' p : ℕ, if p.Prime then ∑' e, f (p ^ e) else 1 = ∑' n, f n`.
This version is stated using `HasProd` and `Set.mulIndicator`. -/
theorem eulerProduct_hasProd_mulIndicator (hsum : Summable (‖f ·‖)) (hf₀ : f 0 = 0) :
    HasProd (Set.mulIndicator {p | Nat.Prime p} fun p ↦  ∑' e, f (p ^ e)) (∑' n, f n) := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    ⊢ HasProd ((setOf fun p => Nat.Prime p).mulIndicator fun p => tsum fun e => f  …
  -/
  rw [← hasProd_subtype_iff_mulIndicator]
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    ⊢ HasProd (Function.comp (fun p => tsum fun e => f (HPow.hPow p e)) Subtype.va …
  -/
  exact eulerProduct_hasProd hf₁ hmul hsum hf₀
  /-
    🎉 no goals
  -/


open Filter in
include hf₁ hmul in
/-- The *Euler Product* for multiplicative (on coprime arguments) functions.

If `f : ℕ → R`, where `R` is a complete normed commutative ring, `f 0 = 0`, `f 1 = 1`, `f` is
multiplicative on coprime arguments, and `‖f ·‖` is summable, then
`∏' p : {p : ℕ | p.Prime}, ∑' e, f (p ^ e) = ∑' n, f n`.
This is a version using convergence of finite partial products. -/
theorem eulerProduct (hsum : Summable (‖f ·‖)) (hf₀ : f 0 = 0) :
    Tendsto (fun n : ℕ ↦ ∏ p ∈ primesBelow n, ∑' e, f (p ^ e)) atTop (𝓝 (∑' n, f n)) := by
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => tsum fun e => f (HPow.h …
  -/
  have := (eulerProduct_hasProd_mulIndicator hf₁ hmul hsum hf₀).tendsto_prod_nat
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    this : Filter.Tendsto (fun n => (Finset.range n).prod fun i => (setOf fun p => …
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => tsum fun e => f (HPow.h …
  -/
  let F : ℕ → R := fun p ↦ ∑' (e : ℕ), f (p ^ e)
  have H (n : ℕ) : ∏ i ∈ range n, Set.mulIndicator {p | Nat.Prime p} F i =
                     ∏ p ∈ primesBelow n, ∑' (e : ℕ), f (p ^ e) :=
    prod_mulIndicator_eq_prod_filter (range n) (fun _ ↦ F) (fun _ ↦ {p | Nat.Prime p}) id
  /-
    R : Type u_1
    inst✝¹ : NormedCommRing R
    f : Nat → R
    inst✝ : CompleteSpace R
    hf₁ : Eq (f 1) 1
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hsum : Summable fun x => Norm.norm (f x)
    hf₀ : Eq (f 0) 0
    this : Filter.Tendsto (fun n => (Finset.range n).prod fun i => (setOf fun p => …
    F : Nat → R := fun p => tsum fun e => f (HPow.hPow p e)
    H : ∀ (n : Nat), Eq ((Finset.range n).prod fun i => (setOf fun p => Nat.Prime  …
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => tsum fun e => f (HPow.h …
  -/
  simpa only [F, H]
  /-
    🎉 no goals
  -/


include hf₁ hmul in
/-- The *Euler Product* for multiplicative (on coprime arguments) functions.

If `f : ℕ → R`, where `R` is a complete normed commutative ring, `f 0 = 0`, `f 1 = 1`, `f` is
multiplicative on coprime arguments, and `‖f ·‖` is summable, then
`∏' p : {p : ℕ | p.Prime}, ∑' e, f (p ^ e) = ∑' n, f n`. -/
theorem eulerProduct_tprod (hsum : Summable (‖f ·‖)) (hf₀ : f 0 = 0) :
    ∏' p : Primes, ∑' e, f (p ^ e) = ∑' n, f n :=
  (eulerProduct_hasProd hf₁ hmul hsum hf₀).tprod_eq


/-- The *Euler Product* for a multiplicative arithmetic function `f` with values in a
complete normed commutative ring `R`: if `‖f ·‖` is summable, then
`∏' p : Nat.Primes, ∑' e, f (p ^ e) = ∑' n, f n`.
This version is stated in terms of `HasProd`. -/
nonrec theorem IsMultiplicative.eulerProduct_hasProd {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) (hsum : Summable (‖f ·‖)) :
    HasProd (fun p : Primes ↦ ∑' e, f (p ^ e)) (∑' n, f n) :=
  eulerProduct_hasProd hf.1 hf.2 hsum f.map_zero


open Filter in
/-- The *Euler Product* for a multiplicative arithmetic function `f` with values in a
complete normed commutative ring `R`: if `‖f ·‖` is summable, then
`∏' p : Nat.Primes, ∑' e, f (p ^ e) = ∑' n, f n`.
This version is stated in the form of convergence of finite partial products. -/
nonrec theorem IsMultiplicative.eulerProduct {f : ArithmeticFunction R} (hf : f.IsMultiplicative)
    (hsum : Summable (‖f ·‖)) :
    Tendsto (fun n : ℕ ↦ ∏ p ∈ primesBelow n, ∑' e, f (p ^ e)) atTop (𝓝 (∑' n, f n)) :=
  eulerProduct hf.1 hf.2 hsum f.map_zero


/-- The *Euler Product* for a multiplicative arithmetic function `f` with values in a
complete normed commutative ring `R`: if `‖f ·‖` is summable, then
`∏' p : Nat.Primes, ∑' e, f (p ^ e) = ∑' n, f n`. -/
nonrec theorem IsMultiplicative.eulerProduct_tprod {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) (hsum : Summable (‖f ·‖)) :
    ∏' p : Primes, ∑' e, f (p ^ e) = ∑' n, f n :=
  eulerProduct_tprod hf.1 hf.2 hsum f.map_zero


lemma one_sub_inv_eq_geometric_of_summable_norm {f : ℕ →*₀ F} {p : ℕ} (hp : p.Prime)
    (hsum : Summable fun x ↦ ‖f x‖) :
    (1 - f p)⁻¹ = ∑' (e : ℕ), f (p ^ e) := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    p : Nat
    hp : Nat.Prime p
    hsum : Summable fun x => Norm.norm (f x)
    ⊢ Eq (Inv.inv (HSub.hSub 1 (f p))) (tsum fun e => f (HPow.hPow p e))
  -/
  simp only [map_pow]
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    p : Nat
    hp : Nat.Prime p
    hsum : Summable fun x => Norm.norm (f x)
    ⊢ Eq (Inv.inv (HSub.hSub 1 (f p))) (tsum fun e => HPow.hPow (f p) e)
  -/
  refine (tsum_geometric_of_norm_lt_one <| summable_geometric_iff_norm_lt_one.mp ?_).symm
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    p : Nat
    hp : Nat.Prime p
    hsum : Summable fun x => Norm.norm (f x)
    ⊢ Summable fun n => HPow.hPow (f p) n
  -/
  refine Summable.of_norm ?_
  simpa only [Function.comp_def, map_pow]
    using hsum.comp_injective <| Nat.pow_right_injective hp.one_lt


/-- Given a (completely) multiplicative function `f : ℕ → F`, where `F` is a normed field,
such that `‖f p‖ < 1` for all primes `p`, we can express the sum of `f n` over all `s`-factored
positive integers `n` as a product of `(1 - f p)⁻¹` over the primes `p ∈ s`. At the same time,
we show that the sum involved converges absolutely. -/
lemma summable_and_hasSum_factoredNumbers_prod_filter_prime_geometric {f : ℕ →* F}
    (h : ∀ {p : ℕ}, p.Prime → ‖f p‖ < 1) (s : Finset ℕ) :
    Summable (fun m : factoredNumbers s ↦ ‖f m‖) ∧
      HasSum (fun m : factoredNumbers s ↦ f m) (∏ p ∈ s with p.Prime, (1 - f p)⁻¹) := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    h : ∀ {p : Nat}, Nat.Prime p → LT.lt (Norm.norm (f p)) 1
    s : Finset Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
  -/
  have hmul {m n} (_ : Nat.Coprime m n) := f.map_mul m n
  have H₁ :
      ∏ p ∈ s with p.Prime, ∑' n : ℕ, f (p ^ n) = ∏ p ∈ s with p.Prime, (1 - f p)⁻¹ := by
    refine prod_congr rfl fun p hp ↦ ?_
    simp only [map_pow]
    exact tsum_geometric_of_norm_lt_one <| h (mem_filter.mp hp).2
  have H₂ : ∀ {p : ℕ}, p.Prime → Summable fun n ↦ ‖f (p ^ n)‖ := by
    intro p hp
    simp only [map_pow]
    refine Summable.of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun _ ↦ norm_pow_le ..) ?_
    exact summable_geometric_iff_norm_lt_one.mpr <| (norm_norm (f p)).symm ▸ h hp
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    h : ∀ {p : Nat}, Nat.Prime p → LT.lt (Norm.norm (f p)) 1
    s : Finset Nat
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    H₁ : Eq ((Finset.filter (fun p => Nat.Prime p) s).prod fun p => tsum fun n =>  …
    H₂ : ∀ {p : Nat}, Nat.Prime p → Summable fun n => Norm.norm (f (HPow.hPow p n))
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
  -/
  exact H₁ ▸ summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum f.map_one hmul H₂ s
  /-
    🎉 no goals
  -/


/-- A version of `EulerProduct.summable_and_hasSum_factoredNumbers_prod_filter_prime_geometric`
in terms of the value of the series. -/
lemma prod_filter_prime_geometric_eq_tsum_factoredNumbers {f : ℕ →* F} (hsum : Summable f)
    (s : Finset ℕ) :
    ∏ p ∈ s with p.Prime, (1 - f p)⁻¹ = ∑' m : factoredNumbers s, f m := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    s : Finset Nat
    ⊢ Eq ((Finset.filter (fun p => Nat.Prime p) s).prod fun p => Inv.inv (HSub.hSu …
  -/
  refine (summable_and_hasSum_factoredNumbers_prod_filter_prime_geometric ?_ s).2.tsum_eq.symm
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    s : Finset Nat
    ⊢ ∀ {p : Nat}, Nat.Prime p → LT.lt (Norm.norm (f p)) 1
  -/
  exact fun {_} hp ↦ hsum.norm_lt_one hp.one_lt
  /-
    🎉 no goals
  -/


/-- Given a (completely) multiplicative function `f : ℕ → F`, where `F` is a normed field,
such that `‖f p‖ < 1` for all primes `p`, we can express the sum of `f n` over all `N`-smooth
positive integers `n` as a product of `(1 - f p)⁻¹` over the primes `p < N`. At the same time,
we show that the sum involved converges absolutely. -/
lemma summable_and_hasSum_smoothNumbers_prod_primesBelow_geometric {f : ℕ →* F}
    (h : ∀ {p : ℕ}, p.Prime → ‖f p‖ < 1) (N : ℕ) :
    Summable (fun m : N.smoothNumbers ↦ ‖f m‖) ∧
      HasSum (fun m : N.smoothNumbers ↦ f m) (∏ p ∈ N.primesBelow, (1 - f p)⁻¹) := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    h : ∀ {p : Nat}, Nat.Prime p → LT.lt (Norm.norm (f p)) 1
    N : Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) (N.primesBe …
  -/
  rw [smoothNumbers_eq_factoredNumbers, primesBelow]
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    h : ∀ {p : Nat}, Nat.Prime p → LT.lt (Norm.norm (f p)) 1
    N : Nat
    ⊢ And (Summable fun m => Norm.norm (f ↑m)) (HasSum (fun m => f ↑m) ((Finset.fi …
  -/
  exact summable_and_hasSum_factoredNumbers_prod_filter_prime_geometric h _
  /-
    🎉 no goals
  -/


/-- A version of `EulerProduct.summable_and_hasSum_smoothNumbers_prod_primesBelow_geometric`
in terms of the value of the series. -/
lemma prod_primesBelow_geometric_eq_tsum_smoothNumbers {f : ℕ →* F} (hsum : Summable f) (N : ℕ) :
    ∏ p ∈ N.primesBelow, (1 - f p)⁻¹ = ∑' m : N.smoothNumbers, f m := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    N : Nat
    ⊢ Eq (N.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (f p))) (tsum fun m =>  …
  -/
  rw [smoothNumbers_eq_factoredNumbers, primesBelow]
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidHom Nat F
    hsum : Summable ⇑f
    N : Nat
    ⊢ Eq ((Finset.filter (fun p => Nat.Prime p) (Finset.range N)).prod fun p => In …
  -/
  exact prod_filter_prime_geometric_eq_tsum_factoredNumbers hsum _
  /-
    🎉 no goals
  -/


/-- The *Euler Product* for completely multiplicative functions.

If `f : ℕ →*₀ F`, where `F` is a complete normed field and `‖f ·‖` is summable, then
`∏' p : Nat.Primes, (1 - f p)⁻¹ = ∑' n, f n`.
This version is stated in terms of `HasProd`. -/
theorem eulerProduct_completely_multiplicative_hasProd {f : ℕ →*₀ F} (hsum : Summable (‖f ·‖)) :
    HasProd (fun p : Primes ↦ (1 - f p)⁻¹) (∑' n, f n) := by
  have H : (fun p : Primes ↦ (1 - f p)⁻¹) = fun p : Primes ↦ ∑' (e : ℕ), f (p ^ e) :=
    funext <| fun p ↦ one_sub_inv_eq_geometric_of_summable_norm p.prop hsum
  simpa only [map_pow, H]
    using eulerProduct_hasProd f.map_one (fun {m n} _ ↦ f.map_mul m n) hsum f.map_zero


/-- The *Euler Product* for completely multiplicative functions.

If `f : ℕ →*₀ F`, where `F` is a complete normed field and `‖f ·‖` is summable, then
`∏' p : Nat.Primes, (1 - f p)⁻¹ = ∑' n, f n`. -/
theorem eulerProduct_completely_multiplicative_tprod {f : ℕ →*₀ F} (hsum : Summable (‖f ·‖)) :
    ∏' p : Primes, (1 - f p)⁻¹ = ∑' n, f n :=
  (eulerProduct_completely_multiplicative_hasProd hsum).tprod_eq


open Filter in
/-- The *Euler Product* for completely multiplicative functions.

If `f : ℕ →*₀ F`, where `F` is a complete normed field and `‖f ·‖` is summable, then
`∏' p : Nat.Primes, (1 - f p)⁻¹ = ∑' n, f n`.
This version is stated in the form of convergence of finite partial products. -/
theorem eulerProduct_completely_multiplicative {f : ℕ →*₀ F} (hsum : Summable (‖f ·‖)) :
    Tendsto (fun n : ℕ ↦ ∏ p ∈ primesBelow n, (1 - f p)⁻¹) atTop (𝓝 (∑' n, f n)) := by
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    hsum : Summable fun x => Norm.norm (f x)
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (f …
  -/
  have hmul {m n} (_ : Nat.Coprime m n) := f.map_mul m n
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    hsum : Summable fun x => Norm.norm (f x)
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (f …
  -/
  have := (eulerProduct_hasProd_mulIndicator f.map_one hmul hsum f.map_zero).tendsto_prod_nat
  have H (n : ℕ) : ∏ p ∈ range n, {p | Nat.Prime p}.mulIndicator (fun p ↦ (1 - f p)⁻¹) p =
                     ∏ p ∈ primesBelow n, (1 - f p)⁻¹ :=
    prod_mulIndicator_eq_prod_filter
      (range n) (fun _ ↦ fun p ↦ (1 - f p)⁻¹) (fun _ ↦ {p | Nat.Prime p}) id
  have H' : {p | Nat.Prime p}.mulIndicator (fun p ↦ (1 - f p)⁻¹) =
              {p | Nat.Prime p}.mulIndicator (fun p ↦ ∑' e : ℕ, f (p ^ e)) :=
    Set.mulIndicator_congr fun p hp ↦ one_sub_inv_eq_geometric_of_summable_norm hp hsum
  /-
    F : Type u_1
    inst✝¹ : NormedField F
    inst✝ : CompleteSpace F
    f : MonoidWithZeroHom Nat F
    hsum : Summable fun x => Norm.norm (f x)
    hmul : ∀ {m n : Nat}, m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    this : Filter.Tendsto (fun n => (Finset.range n).prod fun i => (setOf fun p => …
    H : ∀ (n : Nat), Eq ((Finset.range n).prod fun p => (setOf fun p => Nat.Prime  …
    H' : Eq ((setOf fun p => Nat.Prime p).mulIndicator fun p => Inv.inv (HSub.hSub …
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (f …
  -/
  simpa only [← H, H'] using this
  /-
    🎉 no goals
  -/


