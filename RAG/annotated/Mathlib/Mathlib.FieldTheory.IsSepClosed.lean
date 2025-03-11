/-- Typeclass for separably closed fields.

To show `Polynomial.Splits p f` for an arbitrary ring homomorphism `f`,
see `IsSepClosed.splits_codomain` and `IsSepClosed.splits_domain`.
-/
class IsSepClosed : Prop where
  splits_of_separable : ∀ p : k[X], p.Separable → (p.Splits <| RingHom.id k)


/-- An algebraically closed field is also separably closed. -/
instance IsSepClosed.of_isAlgClosed [IsAlgClosed k] : IsSepClosed k :=
  ⟨fun p _ ↦ IsAlgClosed.splits p⟩


/-- Every separable polynomial splits in the field extension `f : k →+* K` if `K` is
separably closed.

See also `IsSepClosed.splits_domain` for the case where `k` is separably closed.
-/
theorem IsSepClosed.splits_codomain [IsSepClosed K] {f : k →+* K}
    (p : k[X]) (h : p.Separable) : p.Splits f := by
  /-
    k : Type u
    inst✝² : Field k
    K : Type v
    inst✝¹ : Field K
    inst✝ : IsSepClosed K
    f : RingHom k K
    p : Polynomial k
    h : p.Separable
    ⊢ Polynomial.Splits f p
  -/
  convert IsSepClosed.splits_of_separable (p.map f) (Separable.map h); simp [splits_map_iff]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- Every separable polynomial splits in the field extension `f : k →+* K` if `k` is
separably closed.

See also `IsSepClosed.splits_codomain` for the case where `k` is separably closed.
-/
theorem IsSepClosed.splits_domain [IsSepClosed k] {f : k →+* K}
    (p : k[X]) (h : p.Separable) : p.Splits f :=
  Polynomial.splits_of_splits_id _ <| IsSepClosed.splits_of_separable _ h


theorem exists_root [IsSepClosed k] (p : k[X]) (hp : p.degree ≠ 0) (hsep : p.Separable) :
    ∃ x, IsRoot p x :=
  exists_root_of_splits _ (IsSepClosed.splits_of_separable p hsep) hp


/-- If `n ≥ 2` equals zero in a separably closed field `k`, `b ≠ 0`,
then there exists `x` in `k` such that `a * x ^ n + b * x + c = 0`. -/
theorem exists_root_C_mul_X_pow_add_C_mul_X_add_C
    [IsSepClosed k] {n : ℕ} (a b c : k) (hn : (n : k) = 0) (hn' : 2 ≤ n) (hb : b ≠ 0) :
    ∃ x, a * x ^ n + b * x + c = 0 := by
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    n : Nat
    a b c : k
    hn : Eq (↑n) 0
    hn' : LE.le 2 n
    hb : Ne b 0
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HPow.hPow x n)) (HMul …
  -/
  let f : k[X] := C a * X ^ n + C b * X + C c
  have hdeg : f.degree ≠ 0 := degree_ne_of_natDegree_ne <| by
    by_cases ha : a = 0
    · suffices f.natDegree = 1 from this ▸ one_ne_zero
      simp_rw [f, ha, map_zero, zero_mul, zero_add]
      compute_degree!
    · suffices f.natDegree = n from this ▸ (lt_of_lt_of_le zero_lt_two hn').ne'
      simp_rw [f]
      have h0 : n ≠ 0 := by linarith only [hn']
      have h1 : n ≠ 1 := by linarith only [hn']
      have : 1 ≤ n := le_trans one_le_two hn'
      compute_degree!
      simp [h0, h1, ha]
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    n : Nat
    a b c : k
    hn : Eq (↑n) 0
    hn' : LE.le 2 n
    hb : Ne b 0
    f : Polynomial k := HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPo …
    hdeg : Ne f.degree 0
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HPow.hPow x n)) (HMul …
  -/
  have hsep : f.Separable := separable_C_mul_X_pow_add_C_mul_X_add_C a b c hn hb.isUnit
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    n : Nat
    a b c : k
    hn : Eq (↑n) 0
    hn' : LE.le 2 n
    hb : Ne b 0
    f : Polynomial k := HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPo …
    hdeg : Ne f.degree 0
    hsep : f.Separable
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HPow.hPow x n)) (HMul …
  -/
  obtain ⟨x, hx⟩ := exists_root f hdeg hsep
  /-
    case intro
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    n : Nat
    a b c : k
    hn : Eq (↑n) 0
    hn' : LE.le 2 n
    hb : Ne b 0
    f : Polynomial k := HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPo …
    hdeg : Ne f.degree 0
    hsep : f.Separable
    x : k
    hx : f.IsRoot x
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HPow.hPow x n)) (HMul …
  -/
  exact ⟨x, by simpa [f] using hx⟩
  /-
    🎉 no goals
  -/


/-- If a separably closed field `k` is of characteristic `p`, `n ≥ 2` is such that `p ∣ n`, `b ≠ 0`,
then there exists `x` in `k` such that `a * x ^ n + b * x + c = 0`. -/
theorem exists_root_C_mul_X_pow_add_C_mul_X_add_C'
    [IsSepClosed k] (p n : ℕ) (a b c : k) [CharP k p] (hn : p ∣ n) (hn' : 2 ≤ n) (hb : b ≠ 0) :
    ∃ x, a * x ^ n + b * x + c = 0 :=
  exists_root_C_mul_X_pow_add_C_mul_X_add_C a b c ((CharP.cast_eq_zero_iff k p n).2 hn) hn' hb


variable (k) in
/-- A separably closed perfect field is also algebraically closed. -/
instance (priority := 100) isAlgClosed_of_perfectField [IsSepClosed k] [PerfectField k] :
    IsAlgClosed k :=
  IsAlgClosed.of_exists_root k fun p _ h ↦ exists_root p ((degree_pos_of_irreducible h).ne')
    (PerfectField.separable_of_irreducible h)


theorem exists_pow_nat_eq [IsSepClosed k] (x : k) (n : ℕ) [hn : NeZero (n : k)] :
    ∃ z, z ^ n = x := by
  have hn' : 0 < n := Nat.pos_of_ne_zero fun h => by
    rw [h, Nat.cast_zero] at hn
    exact hn.out rfl
  have : degree (X ^ n - C x) ≠ 0 := by
    rw [degree_X_pow_sub_C hn' x]
    exact (WithBot.coe_lt_coe.2 hn').ne'
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    x : k
    n : Nat
    hn : NeZero ↑n
    hn' : LT.lt 0 n
    this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
    ⊢ Exists fun z => Eq (HPow.hPow z n) x
  -/
  by_cases hx : x = 0
    /-
      case pos
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      x : k
      n : Nat
      hn : NeZero ↑n
      hn' : LT.lt 0 n
      this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
      hx : Eq x 0
      ⊢ Exists fun z => Eq (HPow.hPow z n) x
    -/
  · exact ⟨0, by rw [hx, pow_eq_zero_iff hn'.ne']⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      x : k
      n : Nat
      hn : NeZero ↑n
      hn' : LT.lt 0 n
      this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
      hx : Not (Eq x 0)
      ⊢ Exists fun z => Eq (HPow.hPow z n) x
    -/
  · obtain ⟨z, hz⟩ := exists_root _ this <| separable_X_pow_sub_C x hn.out hx
    /-
      case neg.intro
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      x : k
      n : Nat
      hn : NeZero ↑n
      hn' : LT.lt 0 n
      this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
      hx : Not (Eq x 0)
      z : k
      hz : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).IsRoot z
      ⊢ Exists fun z => Eq (HPow.hPow z n) x
    -/
    use z
    /-
      case h
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      x : k
      n : Nat
      hn : NeZero ↑n
      hn' : LT.lt 0 n
      this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
      hx : Not (Eq x 0)
      z : k
      hz : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).IsRoot z
      ⊢ Eq (HPow.hPow z n) x
    -/
    simpa [eval_C, eval_X, eval_pow, eval_sub, IsRoot.def, sub_eq_zero] using hz
    /-
      🎉 no goals
    -/


theorem exists_eq_mul_self [IsSepClosed k] (x : k) [h2 : NeZero (2 : k)] : ∃ z, x = z * z := by
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    x : k
    h2 : NeZero 2
    ⊢ Exists fun z => Eq x (HMul.hMul z z)
  -/
  rcases exists_pow_nat_eq x 2 with ⟨z, rfl⟩
  /-
    case intro
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    h2 : NeZero 2
    z : k
    ⊢ Exists fun z_1 => Eq (HPow.hPow z 2) (HMul.hMul z_1 z_1)
  -/
  exact ⟨z, sq z⟩
  /-
    🎉 no goals
  -/


theorem roots_eq_zero_iff [IsSepClosed k] {p : k[X]} (hsep : p.Separable) :
    p.roots = 0 ↔ p = Polynomial.C (p.coeff 0) := by
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    p : Polynomial k
    hsep : p.Separable
    ⊢ Iff (Eq p.roots 0) (Eq p (Polynomial.C (p.coeff 0)))
  -/
  refine ⟨fun h => ?_, fun hp => by rw [hp, roots_C]⟩
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsSepClosed k
    p : Polynomial k
    hsep : p.Separable
    h : Eq p.roots 0
    ⊢ Eq p (Polynomial.C (p.coeff 0))
  -/
  rcases le_or_lt (degree p) 0 with hd | hd
    /-
      case inl
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      p : Polynomial k
      hsep : p.Separable
      h : Eq p.roots 0
      hd : LE.le p.degree 0
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
  · exact eq_C_of_degree_le_zero hd
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      p : Polynomial k
      hsep : p.Separable
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
  · obtain ⟨z, hz⟩ := IsSepClosed.exists_root p hd.ne' hsep
    /-
      case inr.intro
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      p : Polynomial k
      hsep : p.Separable
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      z : k
      hz : p.IsRoot z
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
    rw [← mem_roots (ne_zero_of_degree_gt hd), h] at hz
    /-
      case inr.intro
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsSepClosed k
      p : Polynomial k
      hsep : p.Separable
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      z : k
      hz : Membership.mem 0 z
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
    simp at hz
    /-
      🎉 no goals
    -/


theorem exists_eval₂_eq_zero [IsSepClosed K] (f : k →+* K)
    (p : k[X]) (hp : p.degree ≠ 0) (hsep : p.Separable) :
    ∃ x, p.eval₂ f x = 0 :=
                                           /-
                                             k : Type u
                                             inst✝² : Field k
                                             K : Type v
                                             inst✝¹ : Field K
                                             inst✝ : IsSepClosed K
                                             f : RingHom k K
                                             p : Polynomial k
                                             hp : Ne p.degree 0
                                             hsep : p.Separable
                                             ⊢ Ne (Polynomial.map f p).degree 0
                                           -/
  let ⟨x, hx⟩ := exists_root (p.map f) (by rwa [degree_map_eq_of_injective f.injective])
                                           /-
                                             🎉 no goals
                                           -/
    (Separable.map hsep)
         /-
           k : Type u
           inst✝² : Field k
           K : Type v
           inst✝¹ : Field K
           inst✝ : IsSepClosed K
           f : RingHom k K
           p : Polynomial k
           hp : Ne p.degree 0
           hsep : p.Separable
           x : K
           hx : (Polynomial.map f p).IsRoot x
           ⊢ Eq (Polynomial.eval₂ f x p) 0
         -/
  ⟨x, by rwa [eval₂_eq_eval_map, ← IsRoot]⟩
         /-
           🎉 no goals
         -/


theorem exists_aeval_eq_zero [IsSepClosed K] [Algebra k K] (p : k[X])
    (hp : p.degree ≠ 0) (hsep : p.Separable) : ∃ x : K, aeval x p = 0 :=
  exists_eval₂_eq_zero (algebraMap k K) p hp hsep


theorem of_exists_root (H : ∀ p : k[X], p.Monic → Irreducible p → Separable p → ∃ x, p.eval x = 0) :
    IsSepClosed k := by
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    ⊢ IsSepClosed k
  -/
  refine ⟨fun p hsep ↦ Or.inr ?_⟩
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    ⊢ ∀ {g : Polynomial k}, Irreducible g → Dvd.dvd g (Polynomial.map (RingHom.id  …
  -/
  intro q hq hdvd
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    q : Polynomial k
    hq : Irreducible q
    hdvd : Dvd.dvd q (Polynomial.map (RingHom.id k) p)
    ⊢ Eq q.degree 1
  -/
  simp only [map_id] at hdvd
  have hlc : IsUnit (leadingCoeff q)⁻¹ := IsUnit.inv <| Ne.isUnit <|
    leadingCoeff_ne_zero.2 <| Irreducible.ne_zero hq
  have hsep' : Separable (q * C (leadingCoeff q)⁻¹) :=
    Separable.mul (Separable.of_dvd hsep hdvd) ((separable_C _).2 hlc)
    (by simpa only [← isCoprime_mul_unit_right_right (isUnit_C.2 hlc) q 1, one_mul]
      using isCoprime_one_right (x := q))
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    q : Polynomial k
    hq : Irreducible q
    hdvd : Dvd.dvd q p
    hlc : IsUnit (Inv.inv q.leadingCoeff)
    hsep' : (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Separable
    ⊢ Eq q.degree 1
  -/
  have hirr' := hq
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    q : Polynomial k
    hq : Irreducible q
    hdvd : Dvd.dvd q p
    hlc : IsUnit (Inv.inv q.leadingCoeff)
    hsep' : (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Separable
    hirr' : Irreducible q
    ⊢ Eq q.degree 1
  -/
  rw [← irreducible_mul_isUnit (isUnit_C.2 hlc)] at hirr'
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    q : Polynomial k
    hq : Irreducible q
    hdvd : Dvd.dvd q p
    hlc : IsUnit (Inv.inv q.leadingCoeff)
    hsep' : (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Separable
    hirr' : Irreducible (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff)))
    ⊢ Eq q.degree 1
  -/
  obtain ⟨x, hx⟩ := H (q * C (leadingCoeff q)⁻¹) (monic_mul_leadingCoeff_inv hq.ne_zero) hirr' hsep'
  /-
    case intro
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → p.Separable → Exists fun x …
    p : Polynomial k
    hsep : p.Separable
    q : Polynomial k
    hq : Irreducible q
    hdvd : Dvd.dvd q p
    hlc : IsUnit (Inv.inv q.leadingCoeff)
    hsep' : (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Separable
    hirr' : Irreducible (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff)))
    x : k
    hx : Eq (Polynomial.eval x (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff) …
    ⊢ Eq q.degree 1
  -/
  exact degree_mul_leadingCoeff_inv q hq.ne_zero ▸ degree_eq_one_of_irreducible_of_root hirr' hx
  /-
    🎉 no goals
  -/


theorem degree_eq_one_of_irreducible [IsSepClosed k] {p : k[X]}
    (hp : Irreducible p) (hsep : p.Separable) : p.degree = 1 :=
  degree_eq_one_of_irreducible_of_splits hp (IsSepClosed.splits_codomain p hsep)


theorem algebraMap_surjective
    [IsSepClosed k] [Algebra k K] [Algebra.IsSeparable k K] :
    Function.Surjective (algebraMap k K) := by
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsSeparable k K
    ⊢ Function.Surjective ⇑(algebraMap k K)
  -/
  refine fun x => ⟨-(minpoly k x).coeff 0, ?_⟩
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsSeparable k K
    x : K
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  have hq : (minpoly k x).leadingCoeff = 1 := minpoly.monic (Algebra.IsSeparable.isIntegral k x)
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsSeparable k K
    x : K
    hq : Eq (minpoly k x).leadingCoeff 1
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  have hsep : IsSeparable k x := Algebra.IsSeparable.isSeparable k x
  have h : (minpoly k x).degree = 1 :=
    degree_eq_one_of_irreducible k (minpoly.irreducible (Algebra.IsSeparable.isIntegral k x)) hsep
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsSeparable k K
    x : K
    hq : Eq (minpoly k x).leadingCoeff 1
    hsep : IsSeparable k x
    h : Eq (minpoly k x).degree 1
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  have : aeval x (minpoly k x) = 0 := minpoly.aeval k x
  rw [eq_X_add_C_of_degree_eq_one h, hq, C_1, one_mul, aeval_add, aeval_X, aeval_C,
    add_eq_zero_iff_eq_neg] at this
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsSeparable k K
    x : K
    hq : Eq (minpoly k x).leadingCoeff 1
    hsep : IsSeparable k x
    h : Eq (minpoly k x).degree 1
    this : Eq x (Neg.neg ((algebraMap k K) ((minpoly k x).coeff 0)))
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  exact (RingHom.map_neg (algebraMap k K) ((minpoly k x).coeff 0)).symm ▸ this.symm
  /-
    🎉 no goals
  -/


/-- If `k` is separably closed, `K / k` is a field extension, `L / k` is an intermediate field
which is separable, then `L` is equal to `k`. A corollary of `IsSepClosed.algebraMap_surjective`. -/
theorem IntermediateField.eq_bot_of_isSepClosed_of_isSeparable [IsSepClosed k] [Algebra k K]
    (L : IntermediateField k K) [Algebra.IsSeparable k L] : L = ⊥ := bot_unique fun x hx ↦ by
  /-
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : Algebra.IsSeparable k (Subtype fun x => Membership.mem L x)
    x : K
    hx : Membership.mem L x
    ⊢ Membership.mem Bot.bot x
  -/
  obtain ⟨y, hy⟩ := IsSepClosed.algebraMap_surjective k L ⟨x, hx⟩
  /-
    case intro
    k : Type u
    inst✝⁴ : Field k
    K : Type v
    inst✝³ : Field K
    inst✝² : IsSepClosed k
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : Algebra.IsSeparable k (Subtype fun x => Membership.mem L x)
    x : K
    hx : Membership.mem L x
    y : k
    hy : Eq ((algebraMap k (Subtype fun x => Membership.mem L x)) y) ⟨x, hx⟩
    ⊢ Membership.mem Bot.bot x
  -/
  exact ⟨y, congr_arg (algebraMap L K) hy⟩
  /-
    🎉 no goals
  -/


/-- Typeclass for an extension being a separable closure. -/
class IsSepClosure [Algebra k K] : Prop where
  sep_closed : IsSepClosed K
  separable : Algebra.IsSeparable k K


/-- A separably closed field is its separable closure. -/
instance IsSepClosure.self_of_isSepClosed [IsSepClosed k] : IsSepClosure k k :=
      /-
        k : Type u
        inst✝² : Field k
        K : Type v
        inst✝¹ : Field K
        inst✝ : IsSepClosed k
        ⊢ IsSepClosed k
      -/
  ⟨by assumption, Algebra.isSeparable_self k⟩
      /-
        🎉 no goals
      -/


/-- If `K` is perfect and is a separable closure of `k`,
then it is also an algebraic closure of `k`. -/
instance (priority := 100) IsSepClosure.isAlgClosure_of_perfectField_top
    [Algebra k K] [IsSepClosure k K] [PerfectField K] : IsAlgClosure k K :=
  haveI : IsSepClosed K := IsSepClosure.sep_closed k
  ⟨inferInstance, IsSepClosure.separable.isAlgebraic⟩


/-- If `k` is perfect, `K` is a separable closure of `k`,
then it is also an algebraic closure of `k`. -/
instance (priority := 100) IsSepClosure.isAlgClosure_of_perfectField
    [Algebra k K] [IsSepClosure k K] [PerfectField k] : IsAlgClosure k K :=
  have halg : Algebra.IsAlgebraic k K := IsSepClosure.separable.isAlgebraic
  haveI := halg.perfectField; inferInstance


/-- If `k` is perfect, `K` is an algebraic closure of `k`,
then it is also a separable closure of `k`. -/
instance (priority := 100) IsSepClosure.of_isAlgClosure_of_perfectField
    [Algebra k K] [IsAlgClosure k K] [PerfectField k] : IsSepClosure k K :=
  ⟨haveI := IsAlgClosure.isAlgClosed (R := k) (K := K); inferInstance,
    (IsAlgClosure.isAlgebraic (R := k) (K := K)).isSeparable_of_perfectField⟩


theorem isSepClosure_iff [Algebra k K] :
    IsSepClosure k K ↔ IsSepClosed K ∧ Algebra.IsSeparable k K :=
  ⟨fun h ↦ ⟨h.1, h.2⟩, fun h ↦ ⟨h.1, h.2⟩⟩


instance isSeparable [Algebra k K] [IsSepClosure k K] : Algebra.IsSeparable k K :=
  IsSepClosure.separable


instance (priority := 100) isGalois [Algebra k K] [IsSepClosure k K] : IsGalois k K where
  to_isSeparable := IsSepClosure.separable
  to_normal.toIsAlgebraic :=  inferInstance
  to_normal.splits' x := (IsSepClosure.sep_closed k).splits_codomain _
    (Algebra.IsSeparable.isSeparable k x)


theorem surjective_restrictDomain_of_isSeparable {E : Type*}
    [Field E] [Algebra K E] [Algebra L E] [IsScalarTower K L E] [Algebra.IsSeparable L E] :
    Function.Surjective fun φ : E →ₐ[K] M ↦ φ.restrictDomain L :=
  fun f ↦ IntermediateField.exists_algHom_of_splits' (E := E) f
    fun s ↦ ⟨Algebra.IsSeparable.isIntegral L s,
      IsSepClosed.splits_codomain _ <| Algebra.IsSeparable.isSeparable L s⟩


@[deprecated (since := "2024-11-15")]
alias surjective_comp_algebraMap_of_isSeparable := surjective_restrictDomain_of_isSeparable


/-- A (random) homomorphism from a separable extension L of K into a separably
  closed extension M of K. -/
noncomputable irreducible_def lift : L →ₐ[K] M :=
  Classical.choice <| IntermediateField.nonempty_algHom_of_adjoin_splits
    (fun x _ ↦ ⟨Algebra.IsSeparable.isIntegral K x,
      splits_codomain _ (Algebra.IsSeparable.isSeparable K x)⟩)
    (IntermediateField.adjoin_univ K L)


/-- A (random) isomorphism between two separable closures of `K`. -/
noncomputable def equiv : L ≃ₐ[K] M :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added to replace local instance above
  haveI : IsSepClosed L := IsSepClosure.sep_closed K
  haveI : IsSepClosed M := IsSepClosure.sep_closed K
  AlgEquiv.ofBijective _ (Normal.toIsAlgebraic.algHom_bijective₂
    (IsSepClosed.lift : L →ₐ[K] M) (IsSepClosed.lift : M →ₐ[K] L)).1


