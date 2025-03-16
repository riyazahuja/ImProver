/-- `I × J` as an ideal of `R × S`. -/
def prod : Ideal (R × S) where
  carrier := { x | x.fst ∈ I ∧ x.snd ∈ J }
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹ : Semiring R
                    inst✝ : Semiring S
                    I : Ideal R
                    J : Ideal S
                    ⊢ Membership.mem { carrier := setOf fun x => And (Membership.mem I x.1) (Membe …
                  -/
  zero_mem' := by simp
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      J : Ideal S
      ⊢ ∀ {a b : Prod R S}, Membership.mem (setOf fun x => And (Membership.mem I x.1 …
    -/
                  /-
                    🎉 no goals
                  -/
    /-
      case mk.mk.intro.intro
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      J : Ideal S
      a₁ : R
      a₂ : S
      b₁ : R
      b₂ : S
      ha₁ : Membership.mem I { fst := a₁, snd := a₂ }.1
      ha₂ : Membership.mem J { fst := a₁, snd := a₂ }.2
      hb₁ : Membership.mem I { fst := b₁, snd := b₂ }.1
      hb₂ : Membership.mem J { fst := b₁, snd := b₂ }.2
      ⊢ Membership.mem (setOf fun x => And (Membership.mem I x.1) (Membership.mem J  …
    -/
  add_mem' := by
    /-
      🎉 no goals
    -/
    rintro ⟨a₁, a₂⟩ ⟨b₁, b₂⟩ ⟨ha₁, ha₂⟩ ⟨hb₁, hb₂⟩
    exact ⟨I.add_mem ha₁ hb₁, J.add_mem ha₂ hb₂⟩
  smul_mem' := by
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      J : Ideal S
      ⊢ ∀ (c : Prod R S) {x : Prod R S}, Membership.mem { carrier := setOf fun x =>  …
    -/
    rintro ⟨a₁, a₂⟩ ⟨b₁, b₂⟩ ⟨hb₁, hb₂⟩
    /-
      case mk.mk.intro
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      J : Ideal S
      a₁ : R
      a₂ : S
      b₁ : R
      b₂ : S
      hb₁ : Membership.mem I { fst := b₁, snd := b₂ }.1
      hb₂ : Membership.mem J { fst := b₁, snd := b₂ }.2
      ⊢ Membership.mem { carrier := setOf fun x => And (Membership.mem I x.1) (Membe …
    -/
    exact ⟨I.mul_mem_left _ hb₁, J.mul_mem_left _ hb₂⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_prod {r : R} {s : S} : (⟨r, s⟩ : R × S) ∈ prod I J ↔ r ∈ I ∧ s ∈ J :=
  Iff.rfl


@[simp]
theorem prod_top_top : prod (⊤ : Ideal R) (⊤ : Ideal S) = ⊤ :=
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹ : Semiring R
                    inst✝ : Semiring S
                    ⊢ ∀ (x : Prod R S), Iff (Membership.mem (Top.top.prod Top.top) x) (Membership. …
                  -/
  Ideal.ext <| by simp
                  /-
                    🎉 no goals
                  -/


/-- Every ideal of the product ring is of the form `I × J`, where `I` and `J` can be explicitly
    given as the image under the projection maps. -/
theorem ideal_prod_eq (I : Ideal (R × S)) :
    I = Ideal.prod (map (RingHom.fst R S) I : Ideal R) (map (RingHom.snd R S) I) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    ⊢ Eq I ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I))
  -/
  apply Ideal.ext
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    ⊢ ∀ (x : Prod R S), Iff (Membership.mem I x) (Membership.mem ((Ideal.map (Ring …
  -/
  rintro ⟨r, s⟩
  rw [mem_prod, mem_map_iff_of_surjective (RingHom.fst R S) Prod.fst_surjective,
    mem_map_iff_of_surjective (RingHom.snd R S) Prod.snd_surjective]
  /-
    case h.mk
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    r : R
    s : S
    ⊢ Iff (Membership.mem I { fst := r, snd := s }) (And (Exists fun x => And (Mem …
  -/
  refine ⟨fun h => ⟨⟨_, ⟨h, rfl⟩⟩, ⟨_, ⟨h, rfl⟩⟩⟩, ?_⟩
  /-
    case h.mk
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    r : R
    s : S
    ⊢ And (Exists fun x => And (Membership.mem I x) (Eq ((RingHom.fst R S) x) r))  …
  -/
  rintro ⟨⟨⟨r, s'⟩, ⟨h₁, rfl⟩⟩, ⟨⟨r', s⟩, ⟨h₂, rfl⟩⟩⟩
  /-
    case h.mk.intro.intro.mk.intro.intro.mk.intro
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    r : R
    s' : S
    h₁ : Membership.mem I { fst := r, snd := s' }
    r' : R
    s : S
    h₂ : Membership.mem I { fst := r', snd := s }
    ⊢ Membership.mem I { fst := (RingHom.fst R S) { fst := r, snd := s' }, snd :=  …
  -/
  simpa using I.add_mem (I.mul_mem_left (1, 0) h₁) (I.mul_mem_left (0, 1) h₂)
  /-
    🎉 no goals
  -/


@[simp]
theorem map_fst_prod (I : Ideal R) (J : Ideal S) : map (RingHom.fst R S) (prod I J) = I := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ Eq (Ideal.map (RingHom.fst R S) (I.prod J)) I
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    x : R
    ⊢ Iff (Membership.mem (Ideal.map (RingHom.fst R S) (I.prod J)) x) (Membership. …
  -/
  rw [mem_map_iff_of_surjective (RingHom.fst R S) Prod.fst_surjective]
  exact
    ⟨by
      rintro ⟨x, ⟨h, rfl⟩⟩
      exact h.1, fun h => ⟨⟨x, 0⟩, ⟨⟨h, Ideal.zero_mem _⟩, rfl⟩⟩⟩


@[simp]
theorem map_snd_prod (I : Ideal R) (J : Ideal S) : map (RingHom.snd R S) (prod I J) = J := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ Eq (Ideal.map (RingHom.snd R S) (I.prod J)) J
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    x : S
    ⊢ Iff (Membership.mem (Ideal.map (RingHom.snd R S) (I.prod J)) x) (Membership. …
  -/
  rw [mem_map_iff_of_surjective (RingHom.snd R S) Prod.snd_surjective]
  exact
    ⟨by
      rintro ⟨x, ⟨h, rfl⟩⟩
      exact h.2, fun h => ⟨⟨0, x⟩, ⟨⟨Ideal.zero_mem _, h⟩, rfl⟩⟩⟩


@[simp]
theorem map_prodComm_prod :
    map ((RingEquiv.prodComm : R × S ≃+* S × R) : R × S →+* S × R) (prod I J) = prod J I := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ Eq (Ideal.map (↑RingEquiv.prodComm) (I.prod J)) (J.prod I)
  -/
  refine Trans.trans (ideal_prod_eq _) ?_
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ Eq ((Ideal.map (RingHom.fst S R) (Ideal.map (↑RingEquiv.prodComm) (I.prod J) …
  -/
  simp [map_map]
  /-
    🎉 no goals
  -/


/-- Ideals of `R × S` are in one-to-one correspondence with pairs of ideals of `R` and ideals of
    `S`. -/
def idealProdEquiv : Ideal (R × S) ≃ Ideal R × Ideal S where
  toFun I := ⟨map (RingHom.fst R S) I, map (RingHom.snd R S) I⟩
  invFun I := prod I.1 I.2
  left_inv I := (ideal_prod_eq I).symm
                                /-
                                  R : Type u
                                  S : Type v
                                  inst✝¹ : Semiring R
                                  inst✝ : Semiring S
                                  I✝ : Ideal R
                                  J✝ : Ideal S
                                  x✝ : Prod (Ideal R) (Ideal S)
                                  I : Ideal R
                                  J : Ideal S
                                  ⊢ Eq ((fun I => { fst := Ideal.map (RingHom.fst R S) I, snd := Ideal.map (Ring …
                                -/
  right_inv := fun ⟨I, J⟩ => by simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem idealProdEquiv_symm_apply (I : Ideal R) (J : Ideal S) :
    idealProdEquiv.symm ⟨I, J⟩ = prod I J :=
  rfl


theorem prod.ext_iff {I I' : Ideal R} {J J' : Ideal S} :
    prod I J = prod I' J' ↔ I = I' ∧ J = J' := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I I' : Ideal R
    J J' : Ideal S
    ⊢ Iff (Eq (I.prod J) (I'.prod J')) (And (Eq I I') (Eq J J'))
  -/
  simp only [← idealProdEquiv_symm_apply, idealProdEquiv.symm.injective.eq_iff, Prod.mk.inj_iff]
  /-
    🎉 no goals
  -/


theorem isPrime_of_isPrime_prod_top {I : Ideal R} (h : (Ideal.prod I (⊤ : Ideal S)).IsPrime) :
    I.IsPrime := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    h : (I.prod Top.top).IsPrime
    ⊢ I.IsPrime
  -/
  constructor
    /-
      case ne_top'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : (I.prod Top.top).IsPrime
      ⊢ Ne I Top.top
    -/
  · contrapose! h
    /-
      case ne_top'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : Eq I Top.top
      ⊢ Not (I.prod Top.top).IsPrime
    -/
    rw [h, prod_top_top, isPrime_iff]
    /-
      case ne_top'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : Eq I Top.top
      ⊢ Not (And (Ne Top.top Top.top) (∀ {x y : Prod R S}, Membership.mem Top.top (H …
    -/
    simp [isPrime_iff, h]
    /-
      🎉 no goals
    -/
    /-
      case mem_or_mem'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : (I.prod Top.top).IsPrime
      ⊢ ∀ {x y : R}, Membership.mem I (HMul.hMul x y) → Or (Membership.mem I x) (Mem …
    -/
  · intro x y hxy
    have : (⟨x, 1⟩ : R × S) * ⟨y, 1⟩ ∈ prod I ⊤ := by
      rw [Prod.mk_mul_mk, mul_one, mem_prod]
      exact ⟨hxy, trivial⟩
    /-
      case mem_or_mem'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : (I.prod Top.top).IsPrime
      x y : R
      hxy : Membership.mem I (HMul.hMul x y)
      this : Membership.mem (I.prod Top.top) (HMul.hMul { fst := x, snd := 1 } { fst …
      ⊢ Or (Membership.mem I x) (Membership.mem I y)
    -/
    simpa using h.mem_or_mem this
    /-
      🎉 no goals
    -/


theorem isPrime_of_isPrime_prod_top' {I : Ideal S} (h : (Ideal.prod (⊤ : Ideal R) I).IsPrime) :
    I.IsPrime := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : (Top.top.prod I).IsPrime
    ⊢ I.IsPrime
  -/
  apply isPrime_of_isPrime_prod_top (S := R)
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : (Top.top.prod I).IsPrime
    ⊢ (I.prod Top.top).IsPrime
  -/
  rw [← map_prodComm_prod]
  -- Note: couldn't synthesize the right instances without the `R` and `S` hints
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : (Top.top.prod I).IsPrime
    ⊢ (Ideal.map (↑RingEquiv.prodComm) (Top.top.prod I)).IsPrime
  -/
  exact map_isPrime_of_equiv (RingEquiv.prodComm (R := R) (S := S))
  /-
    🎉 no goals
  -/


theorem isPrime_ideal_prod_top {I : Ideal R} [h : I.IsPrime] : (prod I (⊤ : Ideal S)).IsPrime := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    h : I.IsPrime
    ⊢ (I.prod Top.top).IsPrime
  -/
  constructor
    /-
      case ne_top'
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : I.IsPrime
      ⊢ Ne (I.prod Top.top) Top.top
    -/
  · rcases h with ⟨h, -⟩
    /-
      case ne_top'.mk
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : Ne I Top.top
      ⊢ Ne (I.prod Top.top) Top.top
    -/
    contrapose! h
    /-
      case ne_top'.mk
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : Eq (I.prod Top.top) Top.top
      ⊢ Eq I Top.top
    -/
    rw [← prod_top_top, prod.ext_iff] at h
    /-
      case ne_top'.mk
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h : And (Eq I Top.top) (Eq Top.top Top.top)
      ⊢ Eq I Top.top
    -/
    exact h.1
    /-
      🎉 no goals
    -/
  /-
    case mem_or_mem'
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    h : I.IsPrime
    ⊢ ∀ {x y : Prod R S}, Membership.mem (I.prod Top.top) (HMul.hMul x y) → Or (Me …
  -/
  rintro ⟨r₁, s₁⟩ ⟨r₂, s₂⟩ ⟨h₁, _⟩
  /-
    case mem_or_mem'.mk.mk.intro
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    h : I.IsPrime
    r₁ : R
    s₁ : S
    r₂ : R
    s₂ : S
    h₁ : Membership.mem I (HMul.hMul { fst := r₁, snd := s₁ } { fst := r₂, snd :=  …
    right✝ : Membership.mem Top.top (HMul.hMul { fst := r₁, snd := s₁ } { fst := r …
    ⊢ Or (Membership.mem (I.prod Top.top) { fst := r₁, snd := s₁ }) (Membership.me …
  -/
  cases' h.mem_or_mem h₁ with h h
    /-
      case mem_or_mem'.mk.mk.intro.inl
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h✝ : I.IsPrime
      r₁ : R
      s₁ : S
      r₂ : R
      s₂ : S
      h₁ : Membership.mem I (HMul.hMul { fst := r₁, snd := s₁ } { fst := r₂, snd :=  …
      right✝ : Membership.mem Top.top (HMul.hMul { fst := r₁, snd := s₁ } { fst := r …
      h : Membership.mem I { fst := r₁, snd := s₁ }.1
      ⊢ Or (Membership.mem (I.prod Top.top) { fst := r₁, snd := s₁ }) (Membership.me …
    -/
  · exact Or.inl ⟨h, trivial⟩
    /-
      🎉 no goals
    -/
    /-
      case mem_or_mem'.mk.mk.intro.inr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal R
      h✝ : I.IsPrime
      r₁ : R
      s₁ : S
      r₂ : R
      s₂ : S
      h₁ : Membership.mem I (HMul.hMul { fst := r₁, snd := s₁ } { fst := r₂, snd :=  …
      right✝ : Membership.mem Top.top (HMul.hMul { fst := r₁, snd := s₁ } { fst := r …
      h : Membership.mem I { fst := r₂, snd := s₂ }.1
      ⊢ Or (Membership.mem (I.prod Top.top) { fst := r₁, snd := s₁ }) (Membership.me …
    -/
  · exact Or.inr ⟨h, trivial⟩
    /-
      🎉 no goals
    -/


theorem isPrime_ideal_prod_top' {I : Ideal S} [h : I.IsPrime] : (prod (⊤ : Ideal R) I).IsPrime := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : I.IsPrime
    ⊢ (Top.top.prod I).IsPrime
  -/
  letI : IsPrime (prod I (⊤ : Ideal R)) := isPrime_ideal_prod_top
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : I.IsPrime
    this : (I.prod Top.top).IsPrime := Ideal.isPrime_ideal_prod_top
    ⊢ (Top.top.prod I).IsPrime
  -/
  rw [← map_prodComm_prod]
  -- Note: couldn't synthesize the right instances without the `R` and `S` hints
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal S
    h : I.IsPrime
    this : (I.prod Top.top).IsPrime := Ideal.isPrime_ideal_prod_top
    ⊢ (Ideal.map (↑RingEquiv.prodComm) (I.prod Top.top)).IsPrime
  -/
  exact map_isPrime_of_equiv (RingEquiv.prodComm (R := S) (S := R))
  /-
    🎉 no goals
  -/


theorem ideal_prod_prime_aux {I : Ideal R} {J : Ideal S} :
    (Ideal.prod I J).IsPrime → I = ⊤ ∨ J = ⊤ := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ (I.prod J).IsPrime → Or (Eq I Top.top) (Eq J Top.top)
  -/
  contrapose!
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ And (Ne I Top.top) (Ne J Top.top) → Not (I.prod J).IsPrime
  -/
  simp only [ne_top_iff_one, isPrime_iff, not_and, not_forall, not_or]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    J : Ideal S
    ⊢ And (Not (Membership.mem I 1)) (Not (Membership.mem J 1)) → Not (Membership. …
  -/
  exact fun ⟨hI, hJ⟩ _ => ⟨⟨0, 1⟩, ⟨1, 0⟩, by simp, by simp [hJ], by simp [hI]⟩
  /-
    🎉 no goals
  -/


/-- Classification of prime ideals in product rings: the prime ideals of `R × S` are precisely the
    ideals of the form `p × S` or `R × p`, where `p` is a prime ideal of `R` or `S`. -/
theorem ideal_prod_prime (I : Ideal (R × S)) :
    I.IsPrime ↔
      (∃ p : Ideal R, p.IsPrime ∧ I = Ideal.prod p ⊤) ∨
        ∃ p : Ideal S, p.IsPrime ∧ I = Ideal.prod ⊤ p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal (Prod R S)
    ⊢ Iff I.IsPrime (Or (Exists fun p => And p.IsPrime (Eq I (p.prod Top.top))) (E …
  -/
  constructor
    /-
      case mp
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal (Prod R S)
      ⊢ I.IsPrime → Or (Exists fun p => And p.IsPrime (Eq I (p.prod Top.top))) (Exis …
    -/
  · rw [ideal_prod_eq I]
    /-
      case mp
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal (Prod R S)
      ⊢ ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).IsPri …
    -/
    intro hI
    /-
      case mp
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal (Prod R S)
      hI : ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).Is …
      ⊢ Or (Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod  …
    -/
    rcases ideal_prod_prime_aux hI with (h | h)
      /-
        case mp.inl
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).Is …
        h : Eq (Ideal.map (RingHom.fst R S) I) Top.top
        ⊢ Or (Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod  …
      -/
    · right
      /-
        case mp.inl.h
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).Is …
        h : Eq (Ideal.map (RingHom.fst R S) I) Top.top
        ⊢ Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod (Ide …
      -/
      rw [h] at hI ⊢
      /-
        case mp.inl.h
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : (Top.top.prod (Ideal.map (RingHom.snd R S) I)).IsPrime
        h : Eq (Ideal.map (RingHom.fst R S) I) Top.top
        ⊢ Exists fun p => And p.IsPrime (Eq (Top.top.prod (Ideal.map (RingHom.snd R S) …
      -/
      exact ⟨_, ⟨isPrime_of_isPrime_prod_top' hI, rfl⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).Is …
        h : Eq (Ideal.map (RingHom.snd R S) I) Top.top
        ⊢ Or (Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod  …
      -/
    · left
      /-
        case mp.inr.h
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : ((Ideal.map (RingHom.fst R S) I).prod (Ideal.map (RingHom.snd R S) I)).Is …
        h : Eq (Ideal.map (RingHom.snd R S) I) Top.top
        ⊢ Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod (Ide …
      -/
      rw [h] at hI ⊢
      /-
        case mp.inr.h
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        I : Ideal (Prod R S)
        hI : ((Ideal.map (RingHom.fst R S) I).prod Top.top).IsPrime
        h : Eq (Ideal.map (RingHom.snd R S) I) Top.top
        ⊢ Exists fun p => And p.IsPrime (Eq ((Ideal.map (RingHom.fst R S) I).prod Top. …
      -/
      exact ⟨_, ⟨isPrime_of_isPrime_prod_top hI, rfl⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      I : Ideal (Prod R S)
      ⊢ Or (Exists fun p => And p.IsPrime (Eq I (p.prod Top.top))) (Exists fun p =>  …
    -/
  · rintro (⟨p, ⟨h, rfl⟩⟩ | ⟨p, ⟨h, rfl⟩⟩)
      /-
        case mpr.inl.intro.intro
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        p : Ideal R
        h : p.IsPrime
        ⊢ (p.prod Top.top).IsPrime
      -/
    · exact isPrime_ideal_prod_top
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro
        R : Type u
        S : Type v
        inst✝¹ : Semiring R
        inst✝ : Semiring S
        p : Ideal S
        h : p.IsPrime
        ⊢ (Top.top.prod p).IsPrime
      -/
    · exact isPrime_ideal_prod_top'
      /-
        🎉 no goals
      -/


