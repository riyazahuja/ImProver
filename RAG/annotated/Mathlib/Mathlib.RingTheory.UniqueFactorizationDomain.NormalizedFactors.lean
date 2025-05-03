local infixl:50 " ~ᵤ " => Associated


/-- Noncomputably determines the multiset of prime factors. -/
noncomputable def normalizedFactors (a : α) : Multiset α :=
  Multiset.map normalize <| factors a


/-- An arbitrary choice of factors of `x : M` is exactly the (unique) normalized set of factors,
if `M` has a trivial group of units. -/
@[simp]
theorem factors_eq_normalizedFactors {M : Type*} [CancelCommMonoidWithZero M]
    [UniqueFactorizationMonoid M] [Subsingleton Mˣ] (x : M) : factors x = normalizedFactors x := by
  /-
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Subsingleton (Units M)
    x : M
    ⊢ Eq (UniqueFactorizationMonoid.factors x) (UniqueFactorizationMonoid.normaliz …
  -/
  unfold normalizedFactors
  /-
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Subsingleton (Units M)
    x : M
    ⊢ Eq (UniqueFactorizationMonoid.factors x) (Multiset.map (⇑normalize) (UniqueF …
  -/
  convert (Multiset.map_id (factors x)).symm
  /-
    case h.e'_3.a.h.e
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Subsingleton (Units M)
    x x✝ : M
    a✝ : Membership.mem (UniqueFactorizationMonoid.factors x) x✝
    ⊢ Eq (⇑normalize) id
  -/
  ext p
  /-
    case h.e'_3.a.h.e.h
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Subsingleton (Units M)
    x x✝ : M
    a✝ : Membership.mem (UniqueFactorizationMonoid.factors x) x✝
    p : M
    ⊢ Eq (normalize p) (id p)
  -/
  exact normalize_eq p
  /-
    🎉 no goals
  -/


theorem prod_normalizedFactors {a : α} (ane0 : a ≠ 0) :
    Associated (normalizedFactors a).prod a := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors a).prod a
  -/
  rw [normalizedFactors, factors, dif_neg ane0]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Associated (Multiset.map (⇑normalize) (Classical.choose ⋯)).prod a
  -/
  refine Associated.trans ?_ (Classical.choose_spec (exists_prime_factors a ane0)).2
  rw [← Associates.mk_eq_mk_iff_associated, ← Associates.prod_mk, ← Associates.prod_mk,
    Multiset.map_map]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Eq (Multiset.map (Function.comp Associates.mk ⇑normalize) (Classical.choose  …
  -/
  congr 2
  /-
    case e_a.e_f
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Eq (Function.comp Associates.mk ⇑normalize) Associates.mk
  -/
  ext
  /-
    case e_a.e_f.h
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    x✝ : α
    ⊢ Eq (Function.comp Associates.mk (⇑normalize) x✝) (Associates.mk x✝)
  -/
  rw [Function.comp_apply, Associates.mk_normalize]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-04")]
alias normalizedFactors_prod := prod_normalizedFactors


theorem prod_normalizedFactors_eq {a : α} (ane0 : a ≠ 0) :
    (normalizedFactors a).prod = normalize a := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors a).prod (normalize a)
  -/
  trans normalize (normalizedFactors a).prod
    /-
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      ane0 : Ne a 0
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors a).prod (normalize (UniqueFa …
    -/
  · rw [normalizedFactors, ← map_multiset_prod, normalize_idem]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      ane0 : Ne a 0
      ⊢ Eq (normalize (UniqueFactorizationMonoid.normalizedFactors a).prod) (normali …
    -/
  · exact normalize_eq_normalize_iff.mpr (dvd_dvd_iff_associated.mpr (prod_normalizedFactors ane0))
    /-
      🎉 no goals
    -/


theorem prime_of_normalized_factor {a : α} : ∀ x : α, x ∈ normalizedFactors a → Prime x := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ ∀ (x : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x  …
  -/
  rw [normalizedFactors, factors]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ ∀ (x : α), Membership.mem (Multiset.map (⇑normalize) (dite (Eq a 0) (fun h = …
  -/
  split_ifs with ane0; · simp
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Not (Eq a 0)
    ⊢ ∀ (x : α), Membership.mem (Multiset.map (fun x => normalize x) (Classical.ch …
  -/
  intro x hx; rcases Multiset.mem_map.1 hx with ⟨y, ⟨hy, rfl⟩⟩
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Not (Eq a 0)
    y : α
    hy : Membership.mem (Classical.choose ⋯) y
    hx : Membership.mem (Multiset.map (fun x => normalize x) (Classical.choose ⋯)) …
    ⊢ Prime (normalize y)
  -/
  rw [(normalize_associated _).prime_iff]
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Not (Eq a 0)
    y : α
    hy : Membership.mem (Classical.choose ⋯) y
    hx : Membership.mem (Multiset.map (fun x => normalize x) (Classical.choose ⋯)) …
    ⊢ Prime y
  -/
  exact (Classical.choose_spec (UniqueFactorizationMonoid.exists_prime_factors a ane0)).1 y hy
  /-
    🎉 no goals
  -/


theorem irreducible_of_normalized_factor {a : α} :
    ∀ x : α, x ∈ normalizedFactors a → Irreducible x := fun x h =>
  (prime_of_normalized_factor x h).irreducible


theorem normalize_normalized_factor {a : α} :
    ∀ x : α, x ∈ normalizedFactors a → normalize x = x := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ ∀ (x : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x  …
  -/
  rw [normalizedFactors, factors]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ ∀ (x : α), Membership.mem (Multiset.map (⇑normalize) (dite (Eq a 0) (fun h = …
  -/
  split_ifs with h; · simp
                      /-
                        🎉 no goals
                      -/
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    h : Not (Eq a 0)
    ⊢ ∀ (x : α), Membership.mem (Multiset.map (fun x => normalize x) (Classical.ch …
  -/
  intro x hx
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    h : Not (Eq a 0)
    x : α
    hx : Membership.mem (Multiset.map (fun x => normalize x) (Classical.choose ⋯)) x
    ⊢ Eq (normalize x) x
  -/
  obtain ⟨y, _, rfl⟩ := Multiset.mem_map.1 hx
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    h : Not (Eq a 0)
    y : α
    left✝ : Membership.mem (Classical.choose ⋯) y
    hx : Membership.mem (Multiset.map (fun x => normalize x) (Classical.choose ⋯)) …
    ⊢ Eq (normalize (normalize y)) (normalize y)
  -/
  apply normalize_idem
  /-
    🎉 no goals
  -/


theorem normalizedFactors_irreducible {a : α} (ha : Irreducible a) :
    normalizedFactors a = {normalize a} := by
  obtain ⟨p, a_assoc, hp⟩ :=
    prime_factors_irreducible ha ⟨prime_of_normalized_factor, prod_normalizedFactors ha.ne_zero⟩
  have p_mem : p ∈ normalizedFactors a := by
    rw [hp]
    exact Multiset.mem_singleton_self _
  /-
    case intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ha : Irreducible a
    p : α
    a_assoc : Associated a p
    hp : Eq (UniqueFactorizationMonoid.normalizedFactors a) (Singleton.singleton p)
    p_mem : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors a) (Singleton.singleton (nor …
  -/
  convert hp
  /-
    case h.e'_3.h.e'_4
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ha : Irreducible a
    p : α
    a_assoc : Associated a p
    hp : Eq (UniqueFactorizationMonoid.normalizedFactors a) (Singleton.singleton p)
    p_mem : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
    ⊢ Eq (normalize a) p
  -/
  rwa [← normalize_normalized_factor p p_mem, normalize_eq_normalize_iff, dvd_dvd_iff_associated]
  /-
    🎉 no goals
  -/


theorem normalizedFactors_eq_of_dvd (a : α) :
    ∀ᵉ (p ∈ normalizedFactors a) (q ∈ normalizedFactors a), p ∣ q → p = q := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ ∀ (p : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p  …
  -/
  intro p hp q hq hdvd
  convert normalize_eq_normalize hdvd
          ((prime_of_normalized_factor _ hp).irreducible.dvd_symm
            (prime_of_normalized_factor _ hq).irreducible hdvd) <;>
    /-
      case h.e'_2
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
      q : α
      hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) q
      hdvd : Dvd.dvd p q
      ⊢ Eq p (normalize p)
    -/
    /-
      🎉 no goals
    -/
    apply (normalize_normalized_factor _ ‹_›).symm
    /-
      🎉 no goals
    -/


theorem exists_mem_normalizedFactors_of_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) :
    p ∣ a → ∃ q ∈ normalizedFactors a, p ~ᵤ q := fun ⟨b, hb⟩ =>
                                    /-
                                      α : Type u_1
                                      inst✝² : CancelCommMonoidWithZero α
                                      inst✝¹ : NormalizationMonoid α
                                      inst✝ : UniqueFactorizationMonoid α
                                      a p : α
                                      ha0 : Ne a 0
                                      hp : Irreducible p
                                      x✝ : Dvd.dvd p a
                                      b : α
                                      hb : Eq a (HMul.hMul p b)
                                      hb0 : Eq b 0
                                      ⊢ False
                                    -/
  have hb0 : b ≠ 0 := fun hb0 => by simp_all
                                    /-
                                      🎉 no goals
                                    -/
  have : Multiset.Rel Associated (p ::ₘ normalizedFactors b) (normalizedFactors a) :=
    factors_unique
      (fun _ hx =>
        (Multiset.mem_cons.1 hx).elim (fun h => h.symm ▸ hp) (irreducible_of_normalized_factor _))
      irreducible_of_normalized_factor
      (Associated.symm <|
        calc
          Multiset.prod (normalizedFactors a) ~ᵤ a := prod_normalizedFactors ha0
          _ = p * b := hb
          _ ~ᵤ Multiset.prod (p ::ₘ normalizedFactors b) := by
            /-
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : UniqueFactorizationMonoid α
              a p : α
              ha0 : Ne a 0
              hp : Irreducible p
              x✝ : Dvd.dvd p a
              b : α
              hb : Eq a (HMul.hMul p b)
              hb0 : Ne b 0
              ⊢ Associated (HMul.hMul p b) (Multiset.cons p (UniqueFactorizationMonoid.norma …
            -/
            rw [Multiset.prod_cons]
            /-
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : UniqueFactorizationMonoid α
              a p : α
              ha0 : Ne a 0
              hp : Irreducible p
              x✝ : Dvd.dvd p a
              b : α
              hb : Eq a (HMul.hMul p b)
              hb0 : Ne b 0
              ⊢ Associated (HMul.hMul p b) (HMul.hMul p (UniqueFactorizationMonoid.normalize …
            -/
            exact (prod_normalizedFactors hb0).symm.mul_left _
            /-
              🎉 no goals
            -/
          )
                                             /-
                                               α : Type u_1
                                               inst✝² : CancelCommMonoidWithZero α
                                               inst✝¹ : NormalizationMonoid α
                                               inst✝ : UniqueFactorizationMonoid α
                                               a p : α
                                               ha0 : Ne a 0
                                               hp : Irreducible p
                                               x✝ : Dvd.dvd p a
                                               b : α
                                               hb : Eq a (HMul.hMul p b)
                                               hb0 : Ne b 0
                                               this : Multiset.Rel Associated (Multiset.cons p (UniqueFactorizationMonoid.nor …
                                               ⊢ Membership.mem (Multiset.cons p (UniqueFactorizationMonoid.normalizedFactors …
                                             -/
  Multiset.exists_mem_of_rel_of_mem this (by simp)
                                             /-
                                               🎉 no goals
                                             -/


theorem exists_mem_normalizedFactors {x : α} (hx : x ≠ 0) (h : ¬IsUnit x) :
    ∃ p, p ∈ normalizedFactors x := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.normalizedFactors  …
  -/
  obtain ⟨p', hp', hp'x⟩ := WfDvdMonoid.exists_irreducible_factor h hx
  /-
    case intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    p' : α
    hp' : Irreducible p'
    hp'x : Dvd.dvd p' x
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.normalizedFactors  …
  -/
  obtain ⟨p, hp, _⟩ := exists_mem_normalizedFactors_of_dvd hx hp' hp'x
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    p' : α
    hp' : Irreducible p'
    hp'x : Dvd.dvd p' x
    p : α
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p
    right✝ : Associated p' p
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.normalizedFactors  …
  -/
  exact ⟨p, hp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem normalizedFactors_zero : normalizedFactors (0 : α) = 0 := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors 0) 0
  -/
  simp [normalizedFactors, factors]
  /-
    🎉 no goals
  -/


@[simp]
theorem normalizedFactors_one : normalizedFactors (1 : α) = 0 := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors 1) 0
  -/
  cases' subsingleton_or_nontrivial α with h h
    /-
      case inl
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      h : Subsingleton α
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors 1) 0
    -/
  · dsimp [normalizedFactors, factors]
    /-
      case inl
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      h : Subsingleton α
      ⊢ Eq (Multiset.map (⇑normalize) (dite (Eq 1 0) (fun h => 0) fun h => Classical …
    -/
    simp [Subsingleton.elim (1 : α) 0]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      h : Nontrivial α
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors 1) 0
    -/
  · rw [← Multiset.rel_zero_right]
    /-
      case inr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      h : Nontrivial α
      ⊢ Multiset.Rel ?m.23386 (UniqueFactorizationMonoid.normalizedFactors 1) 0
    -/
    apply factors_unique irreducible_of_normalized_factor
      /-
        case inr.hg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : UniqueFactorizationMonoid α
        h : Nontrivial α
        ⊢ ∀ (x : α), Membership.mem 0 x → Irreducible x
      -/
    · intro x hx
      /-
        case inr.hg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : UniqueFactorizationMonoid α
        h : Nontrivial α
        x : α
        hx : Membership.mem 0 x
        ⊢ Irreducible x
      -/
      exfalso
      /-
        case inr.hg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : UniqueFactorizationMonoid α
        h : Nontrivial α
        x : α
        hx : Membership.mem 0 x
        ⊢ False
      -/
      apply Multiset.not_mem_zero x hx
      /-
        🎉 no goals
      -/
      /-
        case inr.h
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : UniqueFactorizationMonoid α
        h : Nontrivial α
        ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors 1).prod (Multiset.pr …
      -/
    · apply prod_normalizedFactors one_ne_zero
      /-
        🎉 no goals
      -/


@[simp]
theorem normalizedFactors_mul {x y : α} (hx : x ≠ 0) (hy : y ≠ 0) :
    normalizedFactors (x * y) = normalizedFactors x + normalizedFactors y := by
  have h : (normalize : α → α) = Associates.out ∘ Associates.mk := by
    ext
    rw [Function.comp_apply, Associates.out_mk]
  rw [← Multiset.map_id' (normalizedFactors (x * y)), ← Multiset.map_id' (normalizedFactors x), ←
    Multiset.map_id' (normalizedFactors y), ← Multiset.map_congr rfl normalize_normalized_factor, ←
    Multiset.map_congr rfl normalize_normalized_factor, ←
    Multiset.map_congr rfl normalize_normalized_factor, ← Multiset.map_add, h, ←
    Multiset.map_map Associates.out, eq_comm, ← Multiset.map_map Associates.out]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
    ⊢ Eq (Multiset.map Associates.out (Multiset.map Associates.mk (HAdd.hAdd (Uniq …
  -/
  refine congr rfl ?_
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
    ⊢ Eq (Multiset.map Associates.mk (HAdd.hAdd (UniqueFactorizationMonoid.normali …
  -/
  apply Multiset.map_mk_eq_map_mk_of_rel
  /-
    case hst
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
    ⊢ Multiset.Rel (⇑(Associated.setoid α)) (HAdd.hAdd (UniqueFactorizationMonoid. …
  -/
  apply factors_unique
    /-
      case hst.hf
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
      ⊢ ∀ (x_1 : α), Membership.mem (HAdd.hAdd (UniqueFactorizationMonoid.normalized …
    -/
  · intro x hx
    /-
      case hst.hf
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x✝ y : α
      hx✝ : Ne x✝ 0
      hy : Ne y 0
      h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
      x : α
      hx : Membership.mem (HAdd.hAdd (UniqueFactorizationMonoid.normalizedFactors x✝ …
      ⊢ Irreducible x
    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    rcases Multiset.mem_add.1 hx with (hx | hx) <;> exact irreducible_of_normalized_factor x hx
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case hst.hg
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
      ⊢ ∀ (x_1 : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors (HM …
    -/
  · exact irreducible_of_normalized_factor
    /-
      🎉 no goals
    -/
    /-
      case hst.h
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (⇑normalize) (Function.comp Associates.out Associates.mk)
      ⊢ Associated (HAdd.hAdd (UniqueFactorizationMonoid.normalizedFactors x) (Uniqu …
    -/
  · rw [Multiset.prod_add]
    exact
      ((prod_normalizedFactors hx).mul_mul (prod_normalizedFactors hy)).trans
        (prod_normalizedFactors (mul_ne_zero hx hy)).symm


@[simp]
theorem normalizedFactors_pow {x : α} (n : ℕ) :
    normalizedFactors (x ^ n) = n • normalizedFactors x := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    n : Nat
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x n)) (HSMul.hSMu …
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x 0)) (HSMul.hSMu …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    n : Nat
    ih : Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x n)) (HSMul.h …
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x (HAdd.hAdd n 1) …
  -/
  by_cases h0 : x = 0
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      n : Nat
      ih : Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x n)) (HSMul.h …
      h0 : Eq x 0
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x (HAdd.hAdd n 1) …
    -/
  · simp [h0, zero_pow n.succ_ne_zero, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    n : Nat
    ih : Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x n)) (HSMul.h …
    h0 : Not (Eq x 0)
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow x (HAdd.hAdd n 1) …
  -/
  rw [pow_succ', succ_nsmul', normalizedFactors_mul h0 (pow_ne_zero _ h0), ih]
  /-
    🎉 no goals
  -/


theorem _root_.Irreducible.normalizedFactors_pow {p : α} (hp : Irreducible p) (k : ℕ) :
    normalizedFactors (p ^ k) = Multiset.replicate k (normalize p) := by
  rw [UniqueFactorizationMonoid.normalizedFactors_pow, normalizedFactors_irreducible hp,
    Multiset.nsmul_singleton]


theorem normalizedFactors_prod_eq (s : Multiset α) (hs : ∀ a ∈ s, Irreducible a) :
    normalizedFactors s.prod = s.map normalize := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    s : Multiset α
    hs : ∀ (a : α), Membership.mem s a → Irreducible a
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors s.prod) (Multiset.map (⇑norm …
  -/
  induction' s using Multiset.induction with a s ih
    /-
      case empty
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      hs : ∀ (a : α), Membership.mem 0 a → Irreducible a
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.prod 0)) (Multiset …
    -/
  · rw [Multiset.prod_zero, normalizedFactors_one, Multiset.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      s : Multiset α
      ih : (∀ (a : α), Membership.mem s a → Irreducible a) → Eq (UniqueFactorization …
      hs : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → Irreducible a_1
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.cons a s).prod) (M …
    -/
  · have ia := hs a (Multiset.mem_cons_self a _)
    /-
      case cons
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      s : Multiset α
      ih : (∀ (a : α), Membership.mem s a → Irreducible a) → Eq (UniqueFactorization …
      hs : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → Irreducible a_1
      ia : Irreducible a
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.cons a s).prod) (M …
    -/
    have ib := fun b h => hs b (Multiset.mem_cons_of_mem h)
    /-
      case cons
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      s : Multiset α
      ih : (∀ (a : α), Membership.mem s a → Irreducible a) → Eq (UniqueFactorization …
      hs : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → Irreducible a_1
      ia : Irreducible a
      ib : ∀ (b : α), Membership.mem s b → Irreducible b
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.cons a s).prod) (M …
    -/
    obtain rfl | ⟨b, hb⟩ := s.empty_or_exists_mem
    · rw [Multiset.cons_zero, Multiset.prod_singleton, Multiset.map_singleton,
        normalizedFactors_irreducible ia]
    /-
      case cons.inr.intro
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      s : Multiset α
      ih : (∀ (a : α), Membership.mem s a → Irreducible a) → Eq (UniqueFactorization …
      hs : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → Irreducible a_1
      ia : Irreducible a
      ib : ∀ (b : α), Membership.mem s b → Irreducible b
      b : α
      hb : Membership.mem s b
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.cons a s).prod) (M …
    -/
    haveI := nontrivial_of_ne b 0 (ib b hb).ne_zero
    rw [Multiset.prod_cons, Multiset.map_cons,
      normalizedFactors_mul ia.ne_zero (Multiset.prod_ne_zero fun h => (ib 0 h).ne_zero rfl),
      normalizedFactors_irreducible ia, ih ib, Multiset.singleton_add]


theorem dvd_iff_normalizedFactors_le_normalizedFactors {x y : α} (hx : x ≠ 0) (hy : y ≠ 0) :
    x ∣ y ↔ normalizedFactors x ≤ normalizedFactors y := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Dvd.dvd x y) (LE.le (UniqueFactorizationMonoid.normalizedFactors x) (Un …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Dvd.dvd x y → LE.le (UniqueFactorizationMonoid.normalizedFactors x) (UniqueF …
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mp.intro
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      c : α
      hy : Ne (HMul.hMul x c) 0
      ⊢ LE.le (UniqueFactorizationMonoid.normalizedFactors x) (UniqueFactorizationMo …
    -/
    simp [hx, right_ne_zero_of_mul hy]
    /-
      🎉 no goals
    -/
  · rw [← (prod_normalizedFactors hx).dvd_iff_dvd_left, ←
      (prod_normalizedFactors hy).dvd_iff_dvd_right]
    /-
      case mpr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le (UniqueFactorizationMonoid.normalizedFactors x) (UniqueFactorizationMo …
    -/
    apply Multiset.prod_dvd_prod_of_le
    /-
      🎉 no goals
    -/


theorem _root_.Associated.normalizedFactors_eq {a b : α} (h : Associated a b) :
    normalizedFactors a = normalizedFactors b := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    h : Associated a b
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors a) (UniqueFactorizationMonoi …
  -/
  unfold normalizedFactors
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    h : Associated a b
    ⊢ Eq (Multiset.map (⇑normalize) (UniqueFactorizationMonoid.factors a)) (Multis …
  -/
  have h' : ⇑(normalize (α := α)) = Associates.out ∘ Associates.mk := funext Associates.out_mk
  rw [h', ← Multiset.map_map, ← Multiset.map_map,
    Associates.rel_associated_iff_map_eq_map.mp (factors_rel_of_associated h)]


theorem associated_iff_normalizedFactors_eq_normalizedFactors {x y : α} (hx : x ≠ 0) (hy : y ≠ 0) :
    x ~ᵤ y ↔ normalizedFactors x = normalizedFactors y :=
  ⟨Associated.normalizedFactors_eq, fun h =>
                                                             /-
                                                               α : Type u_1
                                                               inst✝² : CancelCommMonoidWithZero α
                                                               inst✝¹ : NormalizationMonoid α
                                                               inst✝ : UniqueFactorizationMonoid α
                                                               x y : α
                                                               hx : Ne x 0
                                                               hy : Ne y 0
                                                               h : Eq (UniqueFactorizationMonoid.normalizedFactors x) (UniqueFactorizationMon …
                                                               ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors x).prod (UniqueFacto …
                                                             -/
    (prod_normalizedFactors hx).symm.trans (_root_.trans (by rw [h]) (prod_normalizedFactors hy))⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem normalizedFactors_of_irreducible_pow {p : α} (hp : Irreducible p) (k : ℕ) :
    normalizedFactors (p ^ k) = Multiset.replicate k (normalize p) := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    p : α
    hp : Irreducible p
    k : Nat
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HPow.hPow p k)) (Multiset.r …
  -/
  rw [normalizedFactors_pow, normalizedFactors_irreducible hp, Multiset.nsmul_singleton]
  /-
    🎉 no goals
  -/


theorem zero_not_mem_normalizedFactors (x : α) : (0 : α) ∉ normalizedFactors x := fun h =>
  Prime.ne_zero (prime_of_normalized_factor _ h) rfl


theorem dvd_of_mem_normalizedFactors {a p : α} (H : p ∈ normalizedFactors a) : p ∣ a := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    H : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
    ⊢ Dvd.dvd p a
  -/
  by_cases hcases : a = 0
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      H : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
      hcases : Eq a 0
      ⊢ Dvd.dvd p a
    -/
  · rw [hcases]
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      H : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
      hcases : Eq a 0
      ⊢ Dvd.dvd p 0
    -/
    exact dvd_zero p
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      H : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
      hcases : Not (Eq a 0)
      ⊢ Dvd.dvd p a
    -/
  · exact dvd_trans (Multiset.dvd_prod H) (Associated.dvd (prod_normalizedFactors hcases))
    /-
      🎉 no goals
    -/


theorem mem_normalizedFactors_iff [Subsingleton αˣ] {p x : α} (hx : x ≠ 0) :
    p ∈ normalizedFactors x ↔ Prime p ∧ p ∣ x := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizationMonoid α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Subsingleton (Units α)
    p x : α
    hx : Ne x 0
    ⊢ Iff (Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p) (And  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p → And (Prim …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      h : Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p
      ⊢ And (Prime p) (Dvd.dvd p x)
    -/
    exact ⟨prime_of_normalized_factor p h, dvd_of_mem_normalizedFactors h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      ⊢ And (Prime p) (Dvd.dvd p x) → Membership.mem (UniqueFactorizationMonoid.norm …
    -/
  · rintro ⟨hprime, hdvd⟩
    /-
      case mpr.intro
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      hprime : Prime p
      hdvd : Dvd.dvd p x
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p
    -/
    obtain ⟨q, hqmem, hqeq⟩ := exists_mem_normalizedFactors_of_dvd hx hprime.irreducible hdvd
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      hprime : Prime p
      hdvd : Dvd.dvd p x
      q : α
      hqmem : Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) q
      hqeq : Associated p q
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p
    -/
    rw [associated_iff_eq] at hqeq
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      p x : α
      hx : Ne x 0
      hprime : Prime p
      hdvd : Dvd.dvd p x
      q : α
      hqmem : Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) q
      hqeq : Eq p q
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors x) p
    -/
    exact hqeq ▸ hqmem
    /-
      🎉 no goals
    -/


theorem exists_associated_prime_pow_of_unique_normalized_factor {p r : α}
    (h : ∀ {m}, m ∈ normalizedFactors r → m = p) (hr : r ≠ 0) : ∃ i : ℕ, Associated (p ^ i) r := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    p r : α
    h : ∀ {m : α}, Membership.mem (UniqueFactorizationMonoid.normalizedFactors r)  …
    hr : Ne r 0
    ⊢ Exists fun i => Associated (HPow.hPow p i) r
  -/
  use (normalizedFactors r).card
  /-
    case h
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    p r : α
    h : ∀ {m : α}, Membership.mem (UniqueFactorizationMonoid.normalizedFactors r)  …
    hr : Ne r 0
    ⊢ Associated (HPow.hPow p (UniqueFactorizationMonoid.normalizedFactors r).card …
  -/
  have := UniqueFactorizationMonoid.prod_normalizedFactors hr
  /-
    case h
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    p r : α
    h : ∀ {m : α}, Membership.mem (UniqueFactorizationMonoid.normalizedFactors r)  …
    hr : Ne r 0
    this : Associated (UniqueFactorizationMonoid.normalizedFactors r).prod r
    ⊢ Associated (HPow.hPow p (UniqueFactorizationMonoid.normalizedFactors r).card …
  -/
  rwa [Multiset.eq_replicate_of_mem fun b => h, Multiset.prod_replicate] at this
  /-
    🎉 no goals
  -/


theorem normalizedFactors_prod_of_prime [Subsingleton αˣ] {m : Multiset α}
    (h : ∀ p ∈ m, Prime p) : normalizedFactors m.prod = m := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizationMonoid α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Subsingleton (Units α)
    m : Multiset α
    h : ∀ (p : α), Membership.mem m p → Prime p
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors m.prod) m
  -/
  cases subsingleton_or_nontrivial α
  · obtain rfl : m = 0 := by
      refine Multiset.eq_zero_of_forall_not_mem fun x hx ↦ ?_
      simpa [Subsingleton.elim x 0] using h x hx
    /-
      case inl
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : NormalizationMonoid α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Subsingleton (Units α)
      h✝ : Subsingleton α
      h : ∀ (p : α), Membership.mem 0 p → Prime p
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.prod 0)) 0
    -/
    simp
    /-
      🎉 no goals
    -/
  · simpa only [← Multiset.rel_eq, ← associated_eq_eq] using
      prime_factors_unique prime_of_normalized_factor h
        (prod_normalizedFactors (m.prod_ne_zero_of_prime h))


theorem mem_normalizedFactors_eq_of_associated {a b c : α} (ha : a ∈ normalizedFactors c)
    (hb : b ∈ normalizedFactors c) (h : Associated a b) : a = b := by
  rw [← normalize_normalized_factor a ha, ← normalize_normalized_factor b hb,
    normalize_eq_normalize_iff]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    a b c : α
    ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors c) a
    hb : Membership.mem (UniqueFactorizationMonoid.normalizedFactors c) b
    h : Associated a b
    ⊢ And (Dvd.dvd a b) (Dvd.dvd b a)
  -/
  exact Associated.dvd_dvd h
  /-
    🎉 no goals
  -/


@[simp]
theorem normalizedFactors_pos (x : α) (hx : x ≠ 0) : 0 < normalizedFactors x ↔ ¬IsUnit x := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    ⊢ Iff (LT.lt 0 (UniqueFactorizationMonoid.normalizedFactors x)) (Not (IsUnit x))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      ⊢ LT.lt 0 (UniqueFactorizationMonoid.normalizedFactors x) → Not (IsUnit x)
    -/
  · intro h hx
    /-
      case mp
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx✝ : Ne x 0
      h : LT.lt 0 (UniqueFactorizationMonoid.normalizedFactors x)
      hx : IsUnit x
      ⊢ False
    -/
    obtain ⟨p, hp⟩ := Multiset.exists_mem_of_ne_zero h.ne'
    exact
      (prime_of_normalized_factor _ hp).not_unit
        (isUnit_of_dvd_unit (dvd_of_mem_normalizedFactors hp) hx)
    /-
      case mpr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      ⊢ Not (IsUnit x) → LT.lt 0 (UniqueFactorizationMonoid.normalizedFactors x)
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      h : Not (IsUnit x)
      ⊢ LT.lt 0 (UniqueFactorizationMonoid.normalizedFactors x)
    -/
    obtain ⟨p, hp⟩ := exists_mem_normalizedFactors hx h
    exact
      bot_lt_iff_ne_bot.mpr
        (mt Multiset.eq_zero_iff_forall_not_mem.mp (not_forall.mpr ⟨p, not_not.mpr hp⟩))


theorem dvdNotUnit_iff_normalizedFactors_lt_normalizedFactors {x y : α} (hx : x ≠ 0) (hy : y ≠ 0) :
    DvdNotUnit x y ↔ normalizedFactors x < normalizedFactors y := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (DvdNotUnit x y) (LT.lt (UniqueFactorizationMonoid.normalizedFactors x)  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      ⊢ DvdNotUnit x y → LT.lt (UniqueFactorizationMonoid.normalizedFactors x) (Uniq …
    -/
  · rintro ⟨_, c, hc, rfl⟩
    simp only [hx, right_ne_zero_of_mul hy, normalizedFactors_mul, Ne, not_false_iff,
      lt_add_iff_pos_right, normalizedFactors_pos, hc]
    /-
      case mpr
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      x y : α
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LT.lt (UniqueFactorizationMonoid.normalizedFactors x) (UniqueFactorizationMo …
    -/
  · intro h
    exact
      dvdNotUnit_of_dvd_of_not_dvd
        ((dvd_iff_normalizedFactors_le_normalizedFactors hx hy).mpr h.le)
        (mt (dvd_iff_normalizedFactors_le_normalizedFactors hy hx).mp h.not_le)


theorem normalizedFactors_multiset_prod (s : Multiset α) (hs : 0 ∉ s) :
    normalizedFactors (s.prod) = (s.map normalizedFactors).sum := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizationMonoid α
    inst✝ : UniqueFactorizationMonoid α
    s : Multiset α
    hs : Not (Membership.mem s 0)
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors s.prod) (Multiset.map Unique …
  -/
  cases subsingleton_or_nontrivial α
  · obtain rfl : s = 0 := by
      apply Multiset.eq_zero_of_forall_not_mem
      intro _
      convert hs
    /-
      case inl
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizationMonoid α
      inst✝ : UniqueFactorizationMonoid α
      h✝ : Subsingleton α
      hs : Not (Membership.mem 0 0)
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Multiset.prod 0)) (Multiset …
    -/
    simp
    /-
      🎉 no goals
    -/
  induction s using Multiset.induction with
  | empty => simp
  | cons _ _ IH =>
    rw [Multiset.prod_cons, Multiset.map_cons, Multiset.sum_cons, normalizedFactors_mul, IH]
    · exact fun h ↦ hs (Multiset.mem_cons_of_mem h)
    · exact fun h ↦ hs (h ▸ Multiset.mem_cons_self _ _)
    · apply Multiset.prod_ne_zero
      exact fun h ↦ hs (Multiset.mem_cons_of_mem h)


open scoped Classical in
/-- Noncomputably defines a `normalizationMonoid` structure on a `UniqueFactorizationMonoid`. -/
protected noncomputable def normalizationMonoid : NormalizationMonoid α :=
  normalizationMonoidOfMonoidHomRightInverse
    { toFun := fun a : Associates α =>
        if a = 0 then 0
        else
          ((normalizedFactors a).map
              (Classical.choose mk_surjective.hasRightInverse : Associates α → α)).prod
                     /-
                       α : Type u_1
                       inst✝¹ : CancelCommMonoidWithZero α
                       inst✝ : UniqueFactorizationMonoid α
                       ⊢ Eq ((fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (UniqueFacto …
                     -/
      map_one' := by nontriviality α; simp
                                      /-
                                        🎉 no goals
                                      -/
      map_mul' := fun x y => by
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : UniqueFactorizationMonoid α
          x y : Associates α
          ⊢ Eq ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (U …
        -/
        by_cases hx : x = 0
          /-
            case pos
            α : Type u_1
            inst✝¹ : CancelCommMonoidWithZero α
            inst✝ : UniqueFactorizationMonoid α
            x y : Associates α
            hx : Eq x 0
            ⊢ Eq ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (U …
          -/
        · simp [hx]
          /-
            🎉 no goals
          -/
        /-
          case neg
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : UniqueFactorizationMonoid α
          x y : Associates α
          hx : Not (Eq x 0)
          ⊢ Eq ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (U …
        -/
        by_cases hy : y = 0
          /-
            case pos
            α : Type u_1
            inst✝¹ : CancelCommMonoidWithZero α
            inst✝ : UniqueFactorizationMonoid α
            x y : Associates α
            hx : Not (Eq x 0)
            hy : Eq y 0
            ⊢ Eq ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (U …
          -/
        · simp [hy]
          /-
            🎉 no goals
          -/
        /-
          case neg
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : UniqueFactorizationMonoid α
          x y : Associates α
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          ⊢ Eq ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classical.choose ⋯) (U …
        -/
        simp [hx, hy] }
        /-
          🎉 no goals
        -/
    (by
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        ⊢ Function.RightInverse (⇑{ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Cl …
      -/
      intro x
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        x : Associates α
        ⊢ Eq (Associates.mk ({ toFun := fun a => ite (Eq a 0) 0 (Multiset.map (Classic …
      -/
      dsimp
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        x : Associates α
        ⊢ Eq (Associates.mk (ite (Eq x 0) 0 (Multiset.map (Classical.choose ⋯) (Unique …
      -/
      by_cases hx : x = 0
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : UniqueFactorizationMonoid α
          x : Associates α
          hx : Eq x 0
          ⊢ Eq (Associates.mk (ite (Eq x 0) 0 (Multiset.map (Classical.choose ⋯) (Unique …
        -/
      · simp [hx]
        /-
          🎉 no goals
        -/
      have h : Associates.mkMonoidHom ∘ Classical.choose mk_surjective.hasRightInverse =
          (id : Associates α → Associates α) := by
        ext x
        rw [Function.comp_apply, mkMonoidHom_apply,
          Classical.choose_spec mk_surjective.hasRightInverse x]
        rfl
      rw [if_neg hx, ← mkMonoidHom_apply, MonoidHom.map_multiset_prod, map_map, h, map_id, ←
        associated_iff_eq]
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        x : Associates α
        hx : Not (Eq x 0)
        h : Eq (Function.comp (⇑Associates.mkMonoidHom) (Classical.choose ⋯)) id
        ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors x).prod x
      -/
      apply prod_normalizedFactors hx)
      /-
        🎉 no goals
      -/


