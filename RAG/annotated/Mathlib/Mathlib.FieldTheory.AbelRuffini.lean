                                                              /-
                                                                F : Type u_1
                                                                inst✝ : Field F
                                                                ⊢ IsSolvable (Polynomial.Gal 0)
                                                              -/
theorem gal_zero_isSolvable : IsSolvable (0 : F[X]).Gal := by infer_instance
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                             /-
                                                               F : Type u_1
                                                               inst✝ : Field F
                                                               ⊢ IsSolvable (Polynomial.Gal 1)
                                                             -/
theorem gal_one_isSolvable : IsSolvable (1 : F[X]).Gal := by infer_instance
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                              /-
                                                                F : Type u_1
                                                                inst✝ : Field F
                                                                x : F
                                                                ⊢ IsSolvable (Polynomial.C x).Gal
                                                              -/
theorem gal_C_isSolvable (x : F) : IsSolvable (C x).Gal := by infer_instance
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                           /-
                                                             F : Type u_1
                                                             inst✝ : Field F
                                                             ⊢ IsSolvable Polynomial.X.Gal
                                                           -/
theorem gal_X_isSolvable : IsSolvable (X : F[X]).Gal := by infer_instance
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                                        /-
                                                                          F : Type u_1
                                                                          inst✝ : Field F
                                                                          x : F
                                                                          ⊢ IsSolvable (HSub.hSub Polynomial.X (Polynomial.C x)).Gal
                                                                        -/
theorem gal_X_sub_C_isSolvable (x : F) : IsSolvable (X - C x).Gal := by infer_instance
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


                                                                           /-
                                                                             F : Type u_1
                                                                             inst✝ : Field F
                                                                             n : Nat
                                                                             ⊢ IsSolvable (HPow.hPow Polynomial.X n).Gal
                                                                           -/
theorem gal_X_pow_isSolvable (n : ℕ) : IsSolvable (X ^ n : F[X]).Gal := by infer_instance
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem gal_mul_isSolvable {p q : F[X]} (_ : IsSolvable p.Gal) (_ : IsSolvable q.Gal) :
    IsSolvable (p * q).Gal :=
  solvable_of_solvable_injective (Gal.restrictProd_injective p q)


theorem gal_prod_isSolvable {s : Multiset F[X]} (hs : ∀ p ∈ s, IsSolvable (Gal p)) :
    IsSolvable s.prod.Gal := by
  /-
    F : Type u_1
    inst✝ : Field F
    s : Multiset (Polynomial F)
    hs : ∀ (p : Polynomial F), Membership.mem s p → IsSolvable p.Gal
    ⊢ IsSolvable s.prod.Gal
  -/
  apply Multiset.induction_on' s
    /-
      case h₁
      F : Type u_1
      inst✝ : Field F
      s : Multiset (Polynomial F)
      hs : ∀ (p : Polynomial F), Membership.mem s p → IsSolvable p.Gal
      ⊢ IsSolvable (Multiset.prod 0).Gal
    -/
  · exact gal_one_isSolvable
    /-
      🎉 no goals
    -/
    /-
      case h₂
      F : Type u_1
      inst✝ : Field F
      s : Multiset (Polynomial F)
      hs : ∀ (p : Polynomial F), Membership.mem s p → IsSolvable p.Gal
      ⊢ ∀ {a : Polynomial F} {s_1 : Multiset (Polynomial F)}, Membership.mem s a → H …
    -/
  · intro p t hps _ ht
    /-
      case h₂
      F : Type u_1
      inst✝ : Field F
      s : Multiset (Polynomial F)
      hs : ∀ (p : Polynomial F), Membership.mem s p → IsSolvable p.Gal
      p : Polynomial F
      t : Multiset (Polynomial F)
      hps : Membership.mem s p
      a✝ : HasSubset.Subset t s
      ht : IsSolvable t.prod.Gal
      ⊢ IsSolvable (Insert.insert p t).prod.Gal
    -/
    rw [Multiset.insert_eq_cons, Multiset.prod_cons]
    /-
      case h₂
      F : Type u_1
      inst✝ : Field F
      s : Multiset (Polynomial F)
      hs : ∀ (p : Polynomial F), Membership.mem s p → IsSolvable p.Gal
      p : Polynomial F
      t : Multiset (Polynomial F)
      hps : Membership.mem s p
      a✝ : HasSubset.Subset t s
      ht : IsSolvable t.prod.Gal
      ⊢ IsSolvable (HMul.hMul p t.prod).Gal
    -/
    exact gal_mul_isSolvable (hs p hps) ht
    /-
      🎉 no goals
    -/


theorem gal_isSolvable_of_splits {p q : F[X]}
    (_ : Fact (p.Splits (algebraMap F q.SplittingField))) (hq : IsSolvable q.Gal) :
    IsSolvable p.Gal :=
  haveI : IsSolvable (q.SplittingField ≃ₐ[F] q.SplittingField) := hq
  solvable_of_surjective (AlgEquiv.restrictNormalHom_surjective q.SplittingField)


theorem gal_isSolvable_tower (p q : F[X]) (hpq : p.Splits (algebraMap F q.SplittingField))
    (hp : IsSolvable p.Gal) (hq : IsSolvable (q.map (algebraMap F p.SplittingField)).Gal) :
    IsSolvable q.Gal := by
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    ⊢ IsSolvable q.Gal
  -/
  let K := p.SplittingField
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    ⊢ IsSolvable q.Gal
  -/
  let L := q.SplittingField
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    L : Type u_1 := q.SplittingField
    ⊢ IsSolvable q.Gal
  -/
  haveI : Fact (p.Splits (algebraMap F L)) := ⟨hpq⟩
  let ϕ : (L ≃ₐ[K] L) ≃* (q.map (algebraMap F K)).Gal :=
    (IsSplittingField.algEquiv L (q.map (algebraMap F K))).autCongr
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    L : Type u_1 := q.SplittingField
    this : Fact (Polynomial.Splits (algebraMap F L) p)
    ϕ : MulEquiv (AlgEquiv K L L) (Polynomial.map (algebraMap F K) q).Gal := (Poly …
    ⊢ IsSolvable q.Gal
  -/
  have ϕ_inj : Function.Injective ϕ.toMonoidHom := ϕ.injective
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    L : Type u_1 := q.SplittingField
    this : Fact (Polynomial.Splits (algebraMap F L) p)
    ϕ : MulEquiv (AlgEquiv K L L) (Polynomial.map (algebraMap F K) q).Gal := (Poly …
    ϕ_inj : Function.Injective ⇑ϕ.toMonoidHom
    ⊢ IsSolvable q.Gal
  -/
  haveI : IsSolvable (K ≃ₐ[F] K) := hp
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    L : Type u_1 := q.SplittingField
    this✝ : Fact (Polynomial.Splits (algebraMap F L) p)
    ϕ : MulEquiv (AlgEquiv K L L) (Polynomial.map (algebraMap F K) q).Gal := (Poly …
    ϕ_inj : Function.Injective ⇑ϕ.toMonoidHom
    this : IsSolvable (AlgEquiv F K K)
    ⊢ IsSolvable q.Gal
  -/
  haveI : IsSolvable (L ≃ₐ[K] L) := solvable_of_solvable_injective ϕ_inj
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Polynomial.Splits (algebraMap F q.SplittingField) p
    hp : IsSolvable p.Gal
    hq : IsSolvable (Polynomial.map (algebraMap F p.SplittingField) q).Gal
    K : Type u_1 := p.SplittingField
    L : Type u_1 := q.SplittingField
    this✝¹ : Fact (Polynomial.Splits (algebraMap F L) p)
    ϕ : MulEquiv (AlgEquiv K L L) (Polynomial.map (algebraMap F K) q).Gal := (Poly …
    ϕ_inj : Function.Injective ⇑ϕ.toMonoidHom
    this✝ : IsSolvable (AlgEquiv F K K)
    this : IsSolvable (AlgEquiv K L L)
    ⊢ IsSolvable q.Gal
  -/
  exact isSolvable_of_isScalarTower F p.SplittingField q.SplittingField
  /-
    🎉 no goals
  -/


theorem gal_X_pow_sub_one_isSolvable (n : ℕ) : IsSolvable (X ^ n - 1 : F[X]).Gal := by
  /-
    F : Type u_1
    inst✝ : Field F
    n : Nat
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
  -/
  by_cases hn : n = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      hn : Eq n 0
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    -/
  · rw [hn, pow_zero, sub_self]
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      hn : Eq n 0
      ⊢ IsSolvable (Polynomial.Gal 0)
    -/
    exact gal_zero_isSolvable
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
  -/
  have hn' : 0 < n := pos_iff_ne_zero.mpr hn
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
  -/
  have hn'' : (X ^ n - 1 : F[X]) ≠ 0 := X_pow_sub_C_ne_zero hn' 1
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
  -/
  apply isSolvable_of_comm
  /-
    case neg.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    ⊢ ∀ (a b : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal), Eq (HMul.hMul a b) ( …
  -/
  intro σ τ
  /-
    case neg.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    ⊢ Eq (HMul.hMul σ τ) (HMul.hMul τ σ)
  -/
  ext a ha
  /-
    case neg.h.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    a : (HSub.hSub (HPow.hPow Polynomial.X n) 1).SplittingField
    ha : Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X n) 1).rootSet (HSub.hS …
    ⊢ Eq ((HMul.hMul σ τ) a) ((HMul.hMul τ σ) a)
  -/
  simp only [mem_rootSet_of_ne hn'', map_sub, aeval_X_pow, aeval_one, sub_eq_zero] at ha
  have key : ∀ σ : (X ^ n - 1 : F[X]).Gal, ∃ m : ℕ, σ a = a ^ m := by
    intro σ
    lift n to ℕ+ using hn'
    exact map_rootsOfUnity_eq_pow_self σ.toAlgHom (rootsOfUnity.mkOfPowEq a ha)
  /-
    case neg.h.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    a : (HSub.hSub (HPow.hPow Polynomial.X n) 1).SplittingField
    ha : Eq (HPow.hPow a n) 1
    key : ∀ (σ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal), Exists fun m => Eq …
    ⊢ Eq ((HMul.hMul σ τ) a) ((HMul.hMul τ σ) a)
  -/
  obtain ⟨c, hc⟩ := key σ
  /-
    case neg.h.h.intro
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    a : (HSub.hSub (HPow.hPow Polynomial.X n) 1).SplittingField
    ha : Eq (HPow.hPow a n) 1
    key : ∀ (σ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal), Exists fun m => Eq …
    c : Nat
    hc : Eq (σ a) (HPow.hPow a c)
    ⊢ Eq ((HMul.hMul σ τ) a) ((HMul.hMul τ σ) a)
  -/
  obtain ⟨d, hd⟩ := key τ
  /-
    case neg.h.h.intro.intro
    F : Type u_1
    inst✝ : Field F
    n : Nat
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    a : (HSub.hSub (HPow.hPow Polynomial.X n) 1).SplittingField
    ha : Eq (HPow.hPow a n) 1
    key : ∀ (σ : (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal), Exists fun m => Eq …
    c : Nat
    hc : Eq (σ a) (HPow.hPow a c)
    d : Nat
    hd : Eq (τ a) (HPow.hPow a d)
    ⊢ Eq ((HMul.hMul σ τ) a) ((HMul.hMul τ σ) a)
  -/
  rw [σ.mul_apply, τ.mul_apply, hc, map_pow, hd, map_pow, hc, ← pow_mul, pow_mul']
  /-
    🎉 no goals
  -/


theorem gal_X_pow_sub_C_isSolvable_aux (n : ℕ) (a : F)
    (h : (X ^ n - 1 : F[X]).Splits (RingHom.id F)) : IsSolvable (X ^ n - C a).Gal := by
  /-
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      a : F
      h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
      ha : Eq a 0
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    -/
  · rw [ha, C_0, sub_zero]
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      a : F
      h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
      ha : Eq a 0
      ⊢ IsSolvable (HPow.hPow Polynomial.X n).Gal
    -/
    exact gal_X_pow_isSolvable n
    /-
      🎉 no goals
    -/
  have ha' : algebraMap F (X ^ n - C a).SplittingField a ≠ 0 :=
    mt ((injective_iff_map_eq_zero _).mp (RingHom.injective _) a) ha
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  by_cases hn : n = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      a : F
      h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
      ha : Not (Eq a 0)
      ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
      hn : Eq n 0
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    -/
  · rw [hn, pow_zero, ← C_1, ← C_sub]
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      a : F
      h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
      ha : Not (Eq a 0)
      ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
      hn : Eq n 0
      ⊢ IsSolvable (Polynomial.C (HSub.hSub 1 a)).Gal
    -/
    exact gal_C_isSolvable (1 - a)
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  have hn' : 0 < n := pos_iff_ne_zero.mpr hn
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  have hn'' : X ^ n - C a ≠ 0 := X_pow_sub_C_ne_zero hn' a
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  have hn''' : (X ^ n - 1 : F[X]) ≠ 0 := X_pow_sub_C_ne_zero hn' 1
  have mem_range : ∀ {c : (X ^ n - C a).SplittingField},
      (c ^ n = 1 → (∃ d, algebraMap F (X ^ n - C a).SplittingField d = c)) := fun {c} hc =>
    RingHom.mem_range.mp (minpoly.mem_range_of_degree_eq_one F c (h.def.resolve_left hn'''
      (minpoly.irreducible ((SplittingField.instNormal (X ^ n - C a)).isIntegral c))
      (minpoly.dvd F c (by rwa [map_id, map_sub, sub_eq_zero, aeval_X_pow, aeval_one]))))
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
  -/
  apply isSolvable_of_comm
  /-
    case neg.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    ⊢ ∀ (a_1 b : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal), Eq  …
  -/
  intro σ τ
  /-
    case neg.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    ⊢ Eq (HMul.hMul σ τ) (HMul.hMul τ σ)
  -/
  ext b hb
  /-
    case neg.h.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    b : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).SplittingField
    hb : Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).r …
    ⊢ Eq ((HMul.hMul σ τ) b) ((HMul.hMul τ σ) b)
  -/
  rw [mem_rootSet_of_ne hn'', map_sub, aeval_X_pow, aeval_C, sub_eq_zero] at hb
  have hb' : b ≠ 0 := by
    intro hb'
    rw [hb', zero_pow hn] at hb
    exact ha' hb.symm
  have key : ∀ σ : (X ^ n - C a).Gal, ∃ c, σ b = b * algebraMap F _ c := by
    intro σ
    have key : (σ b / b) ^ n = 1 := by rw [div_pow, ← map_pow, hb, σ.commutes, div_self ha']
    obtain ⟨c, hc⟩ := mem_range key
    use c
    rw [hc, mul_div_cancel₀ (σ b) hb']
  /-
    case neg.h.h
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    b : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).SplittingField
    hb : Eq (HPow.hPow b n) ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) ( …
    hb' : Ne b 0
    key : ∀ (σ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal), Exi …
    ⊢ Eq ((HMul.hMul σ τ) b) ((HMul.hMul τ σ) b)
  -/
  obtain ⟨c, hc⟩ := key σ
  /-
    case neg.h.h.intro
    F : Type u_1
    inst✝ : Field F
    n : Nat
    a : F
    h : Polynomial.Splits (RingHom.id F) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ha : Not (Eq a 0)
    ha' : Ne ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
    hn''' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0
    mem_range : ∀ {c : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Spl …
    σ τ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal
    b : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).SplittingField
    hb : Eq (HPow.hPow b n) ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) ( …
    hb' : Ne b 0
    key : ∀ (σ : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Gal), Exi …
    c : F
    hc : Eq (σ b) (HMul.hMul b ((algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n …
    ⊢ Eq ((HMul.hMul σ τ) b) ((HMul.hMul τ σ) b)
  -/
  obtain ⟨d, hd⟩ := key τ
  rw [σ.mul_apply, τ.mul_apply, hc, map_mul, τ.commutes, hd, map_mul, σ.commutes, hc,
    mul_assoc, mul_assoc, mul_right_inj' hb', mul_comm]


theorem splits_X_pow_sub_one_of_X_pow_sub_C {F : Type*} [Field F] {E : Type*} [Field E]
    (i : F →+* E) (n : ℕ) {a : F} (ha : a ≠ 0) (h : (X ^ n - C a).Splits i) :
    (X ^ n - 1 : F[X]).Splits i := by
  /-
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  have ha' : i a ≠ 0 := mt ((injective_iff_map_eq_zero i).mp i.injective a) ha
  /-
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  by_cases hn : n = 0
    /-
      case pos
      F : Type u_3
      inst✝¹ : Field F
      E : Type u_4
      inst✝ : Field E
      i : RingHom F E
      n : Nat
      a : F
      ha : Ne a 0
      h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      ha' : Ne (i a) 0
      hn : Eq n 0
      ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    -/
  · rw [hn, pow_zero, sub_self]
    /-
      case pos
      F : Type u_3
      inst✝¹ : Field F
      E : Type u_4
      inst✝ : Field E
      i : RingHom F E
      n : Nat
      a : F
      ha : Ne a 0
      h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      ha' : Ne (i a) 0
      hn : Eq n 0
      ⊢ Polynomial.Splits i 0
    -/
    exact splits_zero i
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  have hn' : 0 < n := pos_iff_ne_zero.mpr hn
  have hn'' : (X ^ n - C a).degree ≠ 0 :=
    ne_of_eq_of_ne (degree_X_pow_sub_C hn' a) (mt WithBot.coe_eq_coe.mp hn)
  /-
    case neg
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  obtain ⟨b, hb⟩ := exists_root_of_splits i h hn''
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (Polynomial.eval₂ i b (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomia …
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  rw [eval₂_sub, eval₂_X_pow, eval₂_C, sub_eq_zero] at hb
  have hb' : b ≠ 0 := by
    intro hb'
    rw [hb', zero_pow hn] at hb
    exact ha' hb.symm
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  let s := ((X ^ n - C a).map i).roots
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  have hs : _ = _ * (s.map _).prod := eq_prod_roots_of_splits h
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    hs : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  rw [leadingCoeff_X_pow_sub_C hn', RingHom.map_one, C_1, one_mul] at hs
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    hs : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  have hs' : Multiset.card s = n := (natDegree_eq_card_roots h).symm.trans natDegree_X_pow_sub_C
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    hs : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    hs' : Eq s.card n
    ⊢ Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  apply @splits_of_exists_multiset F E _ _ i (X ^ n - 1) (s.map fun c : E => c / b)
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    hs : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    hs' : Eq s.card n
    ⊢ Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) 1)) (HMul.hMul (P …
  -/
  rw [leadingCoeff_X_pow_sub_one hn', RingHom.map_one, C_1, one_mul, Multiset.map_map]
  have C_mul_C : C (i a⁻¹) * C (i a) = 1 := by
    rw [← C_mul, ← i.map_mul, inv_mul_cancel₀ ha, i.map_one, C_1]
  have key1 : (X ^ n - 1 : F[X]).map i = C (i a⁻¹) * ((X ^ n - C a).map i).comp (C b * X) := by
    rw [Polynomial.map_sub, Polynomial.map_sub, Polynomial.map_pow, map_X, map_C,
      Polynomial.map_one, sub_comp, pow_comp, X_comp, C_comp, mul_pow, ← C_pow, hb, mul_sub, ←
      mul_assoc, C_mul_C, one_mul]
  have key2 : ((fun q : E[X] => q.comp (C b * X)) ∘ fun c : E => X - C c) = fun c : E =>
      C b * (X - C (c / b)) := by
    ext1 c
    dsimp only [Function.comp_apply]
    rw [sub_comp, X_comp, C_comp, mul_sub, ← C_mul, mul_div_cancel₀ c hb']
  rw [key1, hs, multiset_prod_comp, Multiset.map_map, key2, Multiset.prod_map_mul,
    -- Porting note: needed for `Multiset.map_const` to work
    show (fun (_ : E) => C b) = Function.const E (C b) by rfl,
    Multiset.map_const, Multiset.prod_replicate, hs', ← C_pow, hb, ← mul_assoc, C_mul_C, one_mul]
  /-
    case neg.intro
    F : Type u_3
    inst✝¹ : Field F
    E : Type u_4
    inst✝ : Field E
    i : RingHom F E
    n : Nat
    a : F
    ha : Ne a 0
    h : Polynomial.Splits i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ha' : Ne (i a) 0
    hn : Not (Eq n 0)
    hn' : LT.lt 0 n
    hn'' : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
    b : E
    hb : Eq (HPow.hPow b n) (i a)
    hb' : Ne b 0
    s : Multiset E := (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
    hs : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    hs' : Eq s.card n
    C_mul_C : Eq (HMul.hMul (Polynomial.C (i (Inv.inv a))) (Polynomial.C (i a))) 1
    key1 : Eq (Polynomial.map i (HSub.hSub (HPow.hPow Polynomial.X n) 1)) (HMul.hM …
    key2 : Eq (Function.comp (fun q => q.comp (HMul.hMul (Polynomial.C b) Polynomi …
    ⊢ Eq (Multiset.map (fun c => HSub.hSub Polynomial.X (Polynomial.C (HDiv.hDiv c …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem gal_X_pow_sub_C_isSolvable (n : ℕ) (x : F) : IsSolvable (X ^ n - C x).Gal := by
  /-
    F : Type u_1
    inst✝ : Field F
    n : Nat
    x : F
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).Gal
  -/
  by_cases hx : x = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Eq x 0
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).Gal
    -/
  · rw [hx, C_0, sub_zero]
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Eq x 0
      ⊢ IsSolvable (HPow.hPow Polynomial.X n).Gal
    -/
    exact gal_X_pow_isSolvable n
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    n : Nat
    x : F
    hx : Not (Eq x 0)
    ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).Gal
  -/
  apply gal_isSolvable_tower (X ^ n - 1) (X ^ n - C x)
    /-
      case neg.hpq
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Not (Eq x 0)
      ⊢ Polynomial.Splits (algebraMap F (HSub.hSub (HPow.hPow Polynomial.X n) (Polyn …
    -/
  · exact splits_X_pow_sub_one_of_X_pow_sub_C _ n hx (SplittingField.splits _)
    /-
      🎉 no goals
    -/
    /-
      case neg.hp
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Not (Eq x 0)
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) 1).Gal
    -/
  · exact gal_X_pow_sub_one_isSolvable n
    /-
      🎉 no goals
    -/
    /-
      case neg.hq
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Not (Eq x 0)
      ⊢ IsSolvable (Polynomial.map (algebraMap F (HSub.hSub (HPow.hPow Polynomial.X  …
    -/
  · rw [Polynomial.map_sub, Polynomial.map_pow, map_X, map_C]
    /-
      case neg.hq
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Not (Eq x 0)
      ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C ((algebraMap  …
    -/
    apply gal_X_pow_sub_C_isSolvable_aux
    /-
      case neg.hq.h
      F : Type u_1
      inst✝ : Field F
      n : Nat
      x : F
      hx : Not (Eq x 0)
      ⊢ Polynomial.Splits (RingHom.id (HSub.hSub (HPow.hPow Polynomial.X n) 1).Split …
    -/
    have key := SplittingField.splits (X ^ n - 1 : F[X])
    rwa [← splits_id_iff_splits, Polynomial.map_sub, Polynomial.map_pow, map_X,
      Polynomial.map_one] at key


/-- Inductive definition of solvable by radicals -/
inductive IsSolvableByRad : E → Prop
  | base (α : F) : IsSolvableByRad (algebraMap F E α)
  | add (α β : E) : IsSolvableByRad α → IsSolvableByRad β → IsSolvableByRad (α + β)
  | neg (α : E) : IsSolvableByRad α → IsSolvableByRad (-α)
  | mul (α β : E) : IsSolvableByRad α → IsSolvableByRad β → IsSolvableByRad (α * β)
  | inv (α : E) : IsSolvableByRad α → IsSolvableByRad α⁻¹
  | rad (α : E) (n : ℕ) (hn : n ≠ 0) : IsSolvableByRad (α ^ n) → IsSolvableByRad α


/-- The intermediate field of solvable-by-radicals elements -/
def solvableByRad : IntermediateField F E where
  carrier := IsSolvableByRad F
  zero_mem' := by
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ Membership.mem { carrier := IsSolvableByRad F, mul_mem' := ⋯, one_mem' := ⋯  …
    -/
    change IsSolvableByRad F 0
                 /-
                   F : Type u_1
                   inst✝² : Field F
                   E : Type u_2
                   inst✝¹ : Field E
                   inst✝ : Algebra F E
                   ⊢ ∀ {a b : E}, Membership.mem { carrier := IsSolvableByRad F, mul_mem' := ⋯, o …
                 -/
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ IsSolvableByRad F 0
    -/
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ Membership.mem { carrier := IsSolvableByRad F, mul_mem' := ⋯ }.carrier 1
    -/
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   inst✝² : Field F
                   E : Type u_2
                   inst✝¹ : Field E
                   inst✝ : Algebra F E
                   ⊢ ∀ {a b : E}, Membership.mem (IsSolvableByRad F) a → Membership.mem (IsSolvab …
                 -/
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ IsSolvableByRad F 1
    -/
                 /-
                   🎉 no goals
                 -/
    convert IsSolvableByRad.base (E := E) (0 : F); rw [RingHom.map_zero]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  add_mem' := by apply IsSolvableByRad.add
  one_mem' := by
    change IsSolvableByRad F 1
    convert IsSolvableByRad.base (E := E) (1 : F); rw [RingHom.map_one]
  mul_mem' := by apply IsSolvableByRad.mul
  inv_mem' := IsSolvableByRad.inv
  algebraMap_mem' := IsSolvableByRad.base


theorem induction (P : solvableByRad F E → Prop)
    (base : ∀ α : F, P (algebraMap F (solvableByRad F E) α))
    (add : ∀ α β : solvableByRad F E, P α → P β → P (α + β))
    (neg : ∀ α : solvableByRad F E, P α → P (-α))
    (mul : ∀ α β : solvableByRad F E, P α → P β → P (α * β))
    (inv : ∀ α : solvableByRad F E, P α → P α⁻¹)
    (rad : ∀ α : solvableByRad F E, ∀ n : ℕ, n ≠ 0 → P (α ^ n) → P α) (α : solvableByRad F E) :
    P α := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
    base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
    add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
    neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
    mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
    inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
    rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
    α : Subtype fun x => Membership.mem (solvableByRad F E) x
    ⊢ P α
  -/
  revert α
  suffices ∀ α : E, IsSolvableByRad F α → ∃ β : solvableByRad F E, ↑β = α ∧ P β by
    intro α
    obtain ⟨α₀, hα₀, Pα⟩ := this α (Subtype.mem α)
    convert Pα
    exact Subtype.ext hα₀.symm
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
    base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
    add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
    neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
    mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
    inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
    rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
    ⊢ ∀ (α : E), IsSolvableByRad F α → Exists fun β => And (Eq (↑β) α) (P β)
  -/
  apply IsSolvableByRad.rec
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α : F), Exists fun β => And (Eq (↑β) ((algebraMap F E) α)) (P β)
    -/
  · exact fun α => ⟨algebraMap F (solvableByRad F E) α, rfl, base α⟩
    /-
      🎉 no goals
    -/
    /-
      case add
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α β : E), IsSolvableByRad F α → IsSolvableByRad F β → (Exists fun β => An …
    -/
  · intro α β _ _ Pα Pβ
    /-
      case add
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α β : E
      a✝¹ : IsSolvableByRad F α
      a✝ : IsSolvableByRad F β
      Pα : Exists fun β => And (Eq (↑β) α) (P β)
      Pβ : Exists fun β_1 => And (Eq (↑β_1) β) (P β_1)
      ⊢ Exists fun β_1 => And (Eq (↑β_1) (HAdd.hAdd α β)) (P β_1)
    -/
    obtain ⟨⟨α₀, hα₀, Pα⟩, β₀, hβ₀, Pβ⟩ := Pα, Pβ
    /-
      case add.intro.intro.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α β : E
      a✝¹ : IsSolvableByRad F α
      a✝ : IsSolvableByRad F β
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) α
      Pα : P α₀
      β₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hβ₀ : Eq (↑β₀) β
      Pβ : P β₀
      ⊢ Exists fun β_1 => And (Eq (↑β_1) (HAdd.hAdd α β)) (P β_1)
    -/
    exact ⟨α₀ + β₀, by rw [← hα₀, ← hβ₀]; rfl, add α₀ β₀ Pα Pβ⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α : E), IsSolvableByRad F α → (Exists fun β => And (Eq (↑β) α) (P β)) → E …
    -/
  · intro α _ Pα
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      a✝ : IsSolvableByRad F α
      Pα : Exists fun β => And (Eq (↑β) α) (P β)
      ⊢ Exists fun β => And (Eq (↑β) (Neg.neg α)) (P β)
    -/
    obtain ⟨α₀, hα₀, Pα⟩ := Pα
    /-
      case neg.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      a✝ : IsSolvableByRad F α
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) α
      Pα : P α₀
      ⊢ Exists fun β => And (Eq (↑β) (Neg.neg α)) (P β)
    -/
    exact ⟨-α₀, by rw [← hα₀]; rfl, neg α₀ Pα⟩
    /-
      🎉 no goals
    -/
    /-
      case mul
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α β : E), IsSolvableByRad F α → IsSolvableByRad F β → (Exists fun β => An …
    -/
  · intro α β _ _ Pα Pβ
    /-
      case mul
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α β : E
      a✝¹ : IsSolvableByRad F α
      a✝ : IsSolvableByRad F β
      Pα : Exists fun β => And (Eq (↑β) α) (P β)
      Pβ : Exists fun β_1 => And (Eq (↑β_1) β) (P β_1)
      ⊢ Exists fun β_1 => And (Eq (↑β_1) (HMul.hMul α β)) (P β_1)
    -/
    obtain ⟨⟨α₀, hα₀, Pα⟩, β₀, hβ₀, Pβ⟩ := Pα, Pβ
    /-
      case mul.intro.intro.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α β : E
      a✝¹ : IsSolvableByRad F α
      a✝ : IsSolvableByRad F β
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) α
      Pα : P α₀
      β₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hβ₀ : Eq (↑β₀) β
      Pβ : P β₀
      ⊢ Exists fun β_1 => And (Eq (↑β_1) (HMul.hMul α β)) (P β_1)
    -/
    exact ⟨α₀ * β₀, by rw [← hα₀, ← hβ₀]; rfl, mul α₀ β₀ Pα Pβ⟩
    /-
      🎉 no goals
    -/
    /-
      case inv
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α : E), IsSolvableByRad F α → (Exists fun β => And (Eq (↑β) α) (P β)) → E …
    -/
  · intro α _ Pα
    /-
      case inv
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      a✝ : IsSolvableByRad F α
      Pα : Exists fun β => And (Eq (↑β) α) (P β)
      ⊢ Exists fun β => And (Eq (↑β) (Inv.inv α)) (P β)
    -/
    obtain ⟨α₀, hα₀, Pα⟩ := Pα
    /-
      case inv.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      a✝ : IsSolvableByRad F α
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) α
      Pα : P α₀
      ⊢ Exists fun β => And (Eq (↑β) (Inv.inv α)) (P β)
    -/
    exact ⟨α₀⁻¹, by rw [← hα₀]; rfl, inv α₀ Pα⟩
    /-
      🎉 no goals
    -/
    /-
      case rad
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      ⊢ ∀ (α : E) (n : Nat), Ne n 0 → IsSolvableByRad F (HPow.hPow α n) → (Exists fu …
    -/
  · intro α n hn hα Pα
    /-
      case rad
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      n : Nat
      hn : Ne n 0
      hα : IsSolvableByRad F (HPow.hPow α n)
      Pα : Exists fun β => And (Eq (↑β) (HPow.hPow α n)) (P β)
      ⊢ Exists fun β => And (Eq (↑β) α) (P β)
    -/
    obtain ⟨α₀, hα₀, Pα⟩ := Pα
    /-
      case rad.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      n : Nat
      hn : Ne n 0
      hα : IsSolvableByRad F (HPow.hPow α n)
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) (HPow.hPow α n)
      Pα : P α₀
      ⊢ Exists fun β => And (Eq (↑β) α) (P β)
    -/
    refine ⟨⟨α, IsSolvableByRad.rad α n hn hα⟩, rfl, rad _ n hn ?_⟩
    /-
      case rad.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      n : Nat
      hn : Ne n 0
      hα : IsSolvableByRad F (HPow.hPow α n)
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) (HPow.hPow α n)
      Pα : P α₀
      ⊢ P (HPow.hPow ⟨α, ⋯⟩ n)
    -/
    convert Pα
    /-
      case h.e'_1
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      P : (Subtype fun x => Membership.mem (solvableByRad F E) x) → Prop
      base : ∀ (α : F), P ((algebraMap F (Subtype fun x => Membership.mem (solvableB …
      add : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      neg : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      mul : ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P …
      inv : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), P α → P ( …
      rad : ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), …
      α : E
      n : Nat
      hn : Ne n 0
      hα : IsSolvableByRad F (HPow.hPow α n)
      α₀ : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα₀ : Eq (↑α₀) (HPow.hPow α n)
      Pα : P α₀
      ⊢ Eq (HPow.hPow ⟨α, ⋯⟩ n) α₀
    -/
    exact Subtype.ext (Eq.trans ((solvableByRad F E).coe_pow _ n) hα₀.symm)
    /-
      🎉 no goals
    -/


theorem isIntegral (α : solvableByRad F E) : IsIntegral F α := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : Subtype fun x => Membership.mem (solvableByRad F E) x
    ⊢ IsIntegral F α
  -/
  revert α
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsIntegral F α
  -/
  apply solvableByRad.induction
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : F), IsIntegral F ((algebraMap F (Subtype fun x => Membership.mem (sol …
    -/
  · exact fun _ => isIntegral_algebraMap
    /-
      🎉 no goals
    -/
    /-
      case add
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), IsIntegral  …
    -/
  · exact fun _ _ => IsIntegral.add
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsIntegral F  …
    -/
  · exact fun _ => IsIntegral.neg
    /-
      🎉 no goals
    -/
    /-
      case mul
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α β : Subtype fun x => Membership.mem (solvableByRad F E) x), IsIntegral  …
    -/
  · exact fun _ _ => IsIntegral.mul
    /-
      🎉 no goals
    -/
    /-
      case inv
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsIntegral F  …
    -/
  · intro α hα
    /-
      case inv
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : Subtype fun x => Membership.mem (solvableByRad F E) x
      hα : IsIntegral F α
      ⊢ IsIntegral F (Inv.inv α)
    -/
    exact IsIntegral.inv hα
    /-
      🎉 no goals
    -/
    /-
      case rad
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), Ne  …
    -/
  · intro α n hn hα
    /-
      case rad
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : Subtype fun x => Membership.mem (solvableByRad F E) x
      n : Nat
      hn : Ne n 0
      hα : IsIntegral F (HPow.hPow α n)
      ⊢ IsIntegral F α
    -/
    obtain ⟨p, h1, h2⟩ := hα.isAlgebraic
    refine IsAlgebraic.isIntegral ⟨p.comp (X ^ n),
      ⟨fun h => h1 (leadingCoeff_eq_zero.mp ?_), by rw [aeval_comp, aeval_X_pow, h2]⟩⟩
    /-
      case rad.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : Subtype fun x => Membership.mem (solvableByRad F E) x
      n : Nat
      hn : Ne n 0
      hα : IsIntegral F (HPow.hPow α n)
      p : Polynomial F
      h1 : Ne p 0
      h2 : Eq ((Polynomial.aeval (HPow.hPow α n)) p) 0
      h : Eq (p.comp (HPow.hPow Polynomial.X n)) 0
      ⊢ Eq p.leadingCoeff 0
    -/
    rwa [← leadingCoeff_eq_zero, leadingCoeff_comp, leadingCoeff_X_pow, one_pow, mul_one] at h
    /-
      case rad.intro.intro
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : Subtype fun x => Membership.mem (solvableByRad F E) x
      n : Nat
      hn : Ne n 0
      hα : IsIntegral F (HPow.hPow α n)
      p : Polynomial F
      h1 : Ne p 0
      h2 : Eq ((Polynomial.aeval (HPow.hPow α n)) p) 0
      h : Eq (p.comp (HPow.hPow Polynomial.X n)).leadingCoeff 0
      ⊢ Ne (HPow.hPow Polynomial.X n).natDegree 0
    -/
    rwa [natDegree_X_pow]
    /-
      🎉 no goals
    -/


/-- The statement to be proved inductively -/
def P (α : solvableByRad F E) : Prop :=
  IsSolvable (minpoly F α).Gal


/-- An auxiliary induction lemma, which is generalized by `solvableByRad.isSolvable`. -/
theorem induction3 {α : solvableByRad F E} {n : ℕ} (hn : n ≠ 0) (hα : P (α ^ n)) : P α := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : Subtype fun x => Membership.mem (solvableByRad F E) x
    n : Nat
    hn : Ne n 0
    hα : solvableByRad.P (HPow.hPow α n)
    ⊢ solvableByRad.P α
  -/
  let p := minpoly F (α ^ n)
  have hp : p.comp (X ^ n) ≠ 0 := by
    intro h
    cases' comp_eq_zero_iff.mp h with h' h'
    · exact minpoly.ne_zero (isIntegral (α ^ n)) h'
    · exact hn (by rw [← @natDegree_C F, ← h'.2, natDegree_X_pow])
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : Subtype fun x => Membership.mem (solvableByRad F E) x
    n : Nat
    hn : Ne n 0
    hα : solvableByRad.P (HPow.hPow α n)
    p : Polynomial F := minpoly F (HPow.hPow α n)
    hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
    ⊢ solvableByRad.P α
  -/
  apply gal_isSolvable_of_splits
  · exact ⟨splits_of_splits_of_dvd _ hp (SplittingField.splits (p.comp (X ^ n)))
      (minpoly.dvd F α (by rw [aeval_comp, aeval_X_pow, minpoly.aeval]))⟩
    /-
      case hq
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : Subtype fun x => Membership.mem (solvableByRad F E) x
      n : Nat
      hn : Ne n 0
      hα : solvableByRad.P (HPow.hPow α n)
      p : Polynomial F := minpoly F (HPow.hPow α n)
      hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
      ⊢ IsSolvable (p.comp (HPow.hPow Polynomial.X n)).Gal
    -/
  · refine gal_isSolvable_tower p (p.comp (X ^ n)) ?_ hα ?_
      /-
        case hq.refine_1
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        ⊢ Polynomial.Splits (algebraMap F (p.comp (HPow.hPow Polynomial.X n)).Splittin …
      -/
    · exact Gal.splits_in_splittingField_of_comp _ _ (by rwa [natDegree_X_pow])
      /-
        🎉 no goals
      -/
      /-
        case hq.refine_2
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        ⊢ IsSolvable (Polynomial.map (algebraMap F p.SplittingField) (p.comp (HPow.hPo …
      -/
    · obtain ⟨s, hs⟩ := (splits_iff_exists_multiset _).1 (SplittingField.splits p)
      /-
        case hq.refine_2.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        ⊢ IsSolvable (Polynomial.map (algebraMap F p.SplittingField) (p.comp (HPow.hPo …
      -/
      rw [map_comp, Polynomial.map_pow, map_X, hs, mul_comp, C_comp]
      /-
        case hq.refine_2.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        ⊢ IsSolvable (HMul.hMul (Polynomial.C ((algebraMap F p.SplittingField) p.leadi …
      -/
      apply gal_mul_isSolvable (gal_C_isSolvable _)
      /-
        case hq.refine_2.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        ⊢ IsSolvable ((Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) …
      -/
      rw [multiset_prod_comp]
      /-
        case hq.refine_2.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        ⊢ IsSolvable (Multiset.map (fun p_1 => p_1.comp (HPow.hPow Polynomial.X n)) (M …
      -/
      apply gal_prod_isSolvable
      /-
        case hq.refine_2.intro.hs
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        ⊢ ∀ (p_1 : Polynomial p.SplittingField), Membership.mem (Multiset.map (fun p_2 …
      -/
      intro q hq
      /-
        case hq.refine_2.intro.hs
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : Polynomial p.SplittingField
        hq : Membership.mem (Multiset.map (fun p_1 => p_1.comp (HPow.hPow Polynomial.X …
        ⊢ IsSolvable q.Gal
      -/
      rw [Multiset.mem_map] at hq
      /-
        case hq.refine_2.intro.hs
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : Polynomial p.SplittingField
        hq : Exists fun a => And (Membership.mem (Multiset.map (fun a => HSub.hSub Pol …
        ⊢ IsSolvable q.Gal
      -/
      obtain ⟨q, hq, rfl⟩ := hq
      /-
        case hq.refine_2.intro.hs.intro.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : Polynomial p.SplittingField
        hq : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
        ⊢ IsSolvable (q.comp (HPow.hPow Polynomial.X n)).Gal
      -/
      rw [Multiset.mem_map] at hq
      /-
        case hq.refine_2.intro.hs.intro.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : Polynomial p.SplittingField
        hq : Exists fun a => And (Membership.mem s a) (Eq (HSub.hSub Polynomial.X (Pol …
        ⊢ IsSolvable (q.comp (HPow.hPow Polynomial.X n)).Gal
      -/
      obtain ⟨q, _, rfl⟩ := hq
      /-
        case hq.refine_2.intro.hs.intro.intro.intro.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : p.SplittingField
        left✝ : Membership.mem s q
        ⊢ IsSolvable ((HSub.hSub Polynomial.X (Polynomial.C q)).comp (HPow.hPow Polyno …
      -/
      rw [sub_comp, X_comp, C_comp]
      /-
        case hq.refine_2.intro.hs.intro.intro.intro.intro
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        α : Subtype fun x => Membership.mem (solvableByRad F E) x
        n : Nat
        hn : Ne n 0
        hα : solvableByRad.P (HPow.hPow α n)
        p : Polynomial F := minpoly F (HPow.hPow α n)
        hp : Ne (p.comp (HPow.hPow Polynomial.X n)) 0
        s : Multiset p.SplittingField
        hs : Eq (Polynomial.map (algebraMap F p.SplittingField) p) (HMul.hMul (Polynom …
        q : p.SplittingField
        left✝ : Membership.mem s q
        ⊢ IsSolvable (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C q)).Gal
      -/
      exact gal_X_pow_sub_C_isSolvable n q
      /-
        🎉 no goals
      -/


/-- An auxiliary induction lemma, which is generalized by `solvableByRad.isSolvable`. -/
theorem induction2 {α β γ : solvableByRad F E} (hγ : γ ∈ F⟮α, β⟯) (hα : P α) (hβ : P β) : P γ := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α β γ : Subtype fun x => Membership.mem (solvableByRad F E) x
    hγ : Membership.mem (IntermediateField.adjoin F (Insert.insert α (Singleton.si …
    hα : solvableByRad.P α
    hβ : solvableByRad.P β
    ⊢ solvableByRad.P γ
  -/
  let p := minpoly F α
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α β γ : Subtype fun x => Membership.mem (solvableByRad F E) x
    hγ : Membership.mem (IntermediateField.adjoin F (Insert.insert α (Singleton.si …
    hα : solvableByRad.P α
    hβ : solvableByRad.P β
    p : Polynomial F := minpoly F α
    ⊢ solvableByRad.P γ
  -/
  let q := minpoly F β
  have hpq := Polynomial.splits_of_splits_mul _
    (mul_ne_zero (minpoly.ne_zero (isIntegral α)) (minpoly.ne_zero (isIntegral β)))
    (SplittingField.splits (p * q))
  let f : ↥F⟮α, β⟯ →ₐ[F] (p * q).SplittingField :=
    Classical.choice <| nonempty_algHom_adjoin_of_splits <| by
      intro x hx
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
      cases hx with rw [hx]
      | inl hx => exact ⟨isIntegral α, hpq.1⟩
      | inr hx => exact ⟨isIntegral β, hpq.2⟩
  have key : minpoly F γ = minpoly F (f ⟨γ, hγ⟩) := by
    refine minpoly.eq_of_irreducible_of_monic
      (minpoly.irreducible (isIntegral γ)) ?_ (minpoly.monic (isIntegral γ))
    suffices aeval (⟨γ, hγ⟩ : F⟮α, β⟯) (minpoly F γ) = 0 by
      rw [aeval_algHom_apply, this, map_zero]
    apply (algebraMap (↥F⟮α, β⟯) (solvableByRad F E)).injective
    simp only [map_zero, _root_.map_eq_zero]
    -- Porting note: end of the proof was `exact minpoly.aeval F γ`.
    apply Subtype.val_injective
    -- This used to be `simp`, but we need `erw` and `simp` after https://github.com/leanprover/lean4/pull/2644
    erw [Polynomial.aeval_subalgebra_coe (minpoly F γ)]
    simp
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α β γ : Subtype fun x => Membership.mem (solvableByRad F E) x
    hγ : Membership.mem (IntermediateField.adjoin F (Insert.insert α (Singleton.si …
    hα : solvableByRad.P α
    hβ : solvableByRad.P β
    p : Polynomial F := minpoly F α
    q : Polynomial F := minpoly F β
    hpq : And (Polynomial.Splits (algebraMap F (HMul.hMul p q).SplittingField) (mi …
    f : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F (Ins …
    key : Eq (minpoly F γ) (minpoly F (f ⟨γ, hγ⟩))
    ⊢ solvableByRad.P γ
  -/
  rw [P, key]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α β γ : Subtype fun x => Membership.mem (solvableByRad F E) x
    hγ : Membership.mem (IntermediateField.adjoin F (Insert.insert α (Singleton.si …
    hα : solvableByRad.P α
    hβ : solvableByRad.P β
    p : Polynomial F := minpoly F α
    q : Polynomial F := minpoly F β
    hpq : And (Polynomial.Splits (algebraMap F (HMul.hMul p q).SplittingField) (mi …
    f : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F (Ins …
    key : Eq (minpoly F γ) (minpoly F (f ⟨γ, hγ⟩))
    ⊢ IsSolvable (minpoly F (f ⟨γ, hγ⟩)).Gal
  -/
  refine gal_isSolvable_of_splits ⟨Normal.splits ?_ (f ⟨γ, hγ⟩)⟩ (gal_mul_isSolvable hα hβ)
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α β γ : Subtype fun x => Membership.mem (solvableByRad F E) x
    hγ : Membership.mem (IntermediateField.adjoin F (Insert.insert α (Singleton.si …
    hα : solvableByRad.P α
    hβ : solvableByRad.P β
    p : Polynomial F := minpoly F α
    q : Polynomial F := minpoly F β
    hpq : And (Polynomial.Splits (algebraMap F (HMul.hMul p q).SplittingField) (mi …
    f : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F (Ins …
    key : Eq (minpoly F γ) (minpoly F (f ⟨γ, hγ⟩))
    ⊢ Normal F (HMul.hMul p q).SplittingField
  -/
  apply SplittingField.instNormal
  /-
    🎉 no goals
  -/


/-- An auxiliary induction lemma, which is generalized by `solvableByRad.isSolvable`. -/
theorem induction1 {α β : solvableByRad F E} (hβ : β ∈ F⟮α⟯) (hα : P α) : P β :=
  induction2 (adjoin.mono F _ _ (ge_of_eq (Set.pair_eq_singleton α)) hβ) hα hα


theorem isSolvable (α : solvableByRad F E) : IsSolvable (minpoly F α).Gal := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : Subtype fun x => Membership.mem (solvableByRad F E) x
    ⊢ IsSolvable (minpoly F α).Gal
  -/
  revert α
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsSolvable (m …
  -/
  apply solvableByRad.induction
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : F), IsSolvable (minpoly F ((algebraMap F (Subtype fun x => Membership …
    -/
  · exact fun α => by rw [minpoly.eq_X_sub_C (solvableByRad F E)]; exact gal_X_sub_C_isSolvable α
    /-
      🎉 no goals
    -/
  · exact fun α β => induction2 (add_mem (subset_adjoin F _ (Set.mem_insert α _))
      (subset_adjoin F _ (Set.mem_insert_of_mem α (Set.mem_singleton β))))
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsSolvable (m …
    -/
  · exact fun α => induction1 (neg_mem (mem_adjoin_simple_self F α))
    /-
      🎉 no goals
    -/
  · exact fun α β => induction2 (mul_mem (subset_adjoin F _ (Set.mem_insert α _))
      (subset_adjoin F _ (Set.mem_insert_of_mem α (Set.mem_singleton β))))
    /-
      case inv
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x), IsSolvable (m …
    -/
  · exact fun α => induction1 (inv_mem (mem_adjoin_simple_self F α))
    /-
      🎉 no goals
    -/
    /-
      case rad
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ ∀ (α : Subtype fun x => Membership.mem (solvableByRad F E) x) (n : Nat), Ne  …
    -/
  · exact fun α n => induction3
    /-
      🎉 no goals
    -/


/-- **Abel-Ruffini Theorem** (one direction): An irreducible polynomial with an
`IsSolvableByRad` root has solvable Galois group -/
theorem isSolvable' {α : E} {q : F[X]} (q_irred : Irreducible q) (q_aeval : aeval α q = 0)
    (hα : IsSolvableByRad F α) : IsSolvable q.Gal := by
  have : _root_.IsSolvable (q * C q.leadingCoeff⁻¹).Gal := by
    rw [minpoly.eq_of_irreducible q_irred q_aeval, ←
      show minpoly F (⟨α, hα⟩ : solvableByRad F E) = minpoly F α from
        (minpoly.algebraMap_eq (RingHom.injective _) _).symm]
    exact isSolvable ⟨α, hα⟩
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    q : Polynomial F
    q_irred : Irreducible q
    q_aeval : Eq ((Polynomial.aeval α) q) 0
    hα : IsSolvableByRad F α
    this : IsSolvable (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Gal
    ⊢ IsSolvable q.Gal
  -/
  refine solvable_of_surjective (Gal.restrictDvd_surjective ⟨C q.leadingCoeff⁻¹, rfl⟩ ?_)
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    q : Polynomial F
    q_irred : Irreducible q
    q_aeval : Eq ((Polynomial.aeval α) q) 0
    hα : IsSolvableByRad F α
    this : IsSolvable (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Gal
    ⊢ Ne (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))) 0
  -/
  rw [mul_ne_zero_iff, Ne, Ne, C_eq_zero, inv_eq_zero]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    q : Polynomial F
    q_irred : Irreducible q
    q_aeval : Eq ((Polynomial.aeval α) q) 0
    hα : IsSolvableByRad F α
    this : IsSolvable (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Gal
    ⊢ And (Not (Eq q 0)) (Not (Eq q.leadingCoeff 0))
  -/
  exact ⟨q_irred.ne_zero, leadingCoeff_ne_zero.mpr q_irred.ne_zero⟩
  /-
    🎉 no goals
  -/


