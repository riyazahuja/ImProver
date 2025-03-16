instance maximalIdeal.isMaximal : (maximalIdeal R).IsMaximal := by
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    ⊢ (IsLocalRing.maximalIdeal R).IsMaximal
  -/
  rw [Ideal.isMaximal_iff]
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    ⊢ And (Not (Membership.mem (IsLocalRing.maximalIdeal R) 1)) (∀ (J : Ideal R) ( …
  -/
  constructor
    /-
      case left
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      ⊢ Not (Membership.mem (IsLocalRing.maximalIdeal R) 1)
    -/
  · intro h
    /-
      case left
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      h : Membership.mem (IsLocalRing.maximalIdeal R) 1
      ⊢ False
    -/
    apply h
    /-
      case left
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      h : Membership.mem (IsLocalRing.maximalIdeal R) 1
      ⊢ IsUnit 1
    -/
    exact isUnit_one
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      ⊢ ∀ (J : Ideal R) (x : R), LE.le (IsLocalRing.maximalIdeal R) J → Not (Members …
    -/
  · intro I x _ hx H
    /-
      case right
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      I : Ideal R
      x : R
      a✝ : LE.le (IsLocalRing.maximalIdeal R) I
      hx : Not (Membership.mem (IsLocalRing.maximalIdeal R) x)
      H : Membership.mem I x
      ⊢ Membership.mem I 1
    -/
    erw [Classical.not_not] at hx
    /-
      case right
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      I : Ideal R
      x : R
      a✝ : LE.le (IsLocalRing.maximalIdeal R) I
      hx : IsUnit x
      H : Membership.mem I x
      ⊢ Membership.mem I 1
    -/
    rcases hx with ⟨u, rfl⟩
    /-
      case right.intro
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      I : Ideal R
      a✝ : LE.le (IsLocalRing.maximalIdeal R) I
      u : Units R
      H : Membership.mem I ↑u
      ⊢ Membership.mem I 1
    -/
    simpa using I.mul_mem_left (↑u⁻¹) H
    /-
      🎉 no goals
    -/


theorem maximal_ideal_unique : ∃! I : Ideal R, I.IsMaximal :=
  ⟨maximalIdeal R, maximalIdeal.isMaximal R, fun I hI =>
    hI.eq_of_le (maximalIdeal.isMaximal R).1.1 fun _ hx => hI.1.1 ∘ I.eq_top_of_isUnit_mem hx⟩


theorem eq_maximalIdeal {I : Ideal R} (hI : I.IsMaximal) : I = maximalIdeal R :=
  ExistsUnique.unique (maximal_ideal_unique R) hI <| maximalIdeal.isMaximal R


theorem le_maximalIdeal {J : Ideal R} (hJ : J ≠ ⊤) : J ≤ maximalIdeal R := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    J : Ideal R
    hJ : Ne J Top.top
    ⊢ LE.le J (IsLocalRing.maximalIdeal R)
  -/
  rcases Ideal.exists_le_maximal J hJ with ⟨M, hM1, hM2⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    J : Ideal R
    hJ : Ne J Top.top
    M : Ideal R
    hM1 : M.IsMaximal
    hM2 : LE.le J M
    ⊢ LE.le J (IsLocalRing.maximalIdeal R)
  -/
  rwa [← eq_maximalIdeal hM1]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_maximalIdeal (x) : x ∈ maximalIdeal R ↔ x ∈ nonunits R :=
  Iff.rfl


/--
An element `x` of a commutative local semiring is not contained in the maximal ideal
iff it is a unit.
-/
theorem not_mem_maximalIdeal {x : R} : x ∉ maximalIdeal R ↔ IsUnit x := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    x : R
    ⊢ Iff (Not (Membership.mem (IsLocalRing.maximalIdeal R) x)) (IsUnit x)
  -/
  simp only [mem_maximalIdeal, mem_nonunits_iff, not_not]
  /-
    🎉 no goals
  -/


theorem isField_iff_maximalIdeal_eq : IsField R ↔ maximalIdeal R = ⊥ :=
  not_iff_not.mp
    ⟨Ring.ne_bot_of_isMaximal_of_not_isField inferInstance, fun h =>
      Ring.not_isField_iff_exists_prime.mpr ⟨_, h, Ideal.IsMaximal.isPrime' _⟩⟩


@[deprecated (since := "2024-11-11")]
alias LocalRing.maximal_ideal_unique := IsLocalRing.maximal_ideal_unique


@[deprecated (since := "2024-11-11")]
alias LocalRing.eq_maximalIdeal := IsLocalRing.eq_maximalIdeal


@[deprecated (since := "2024-11-11")]
alias LocalRing.le_maximalIdeal := IsLocalRing.le_maximalIdeal


@[deprecated (since := "2024-11-11")]
alias LocalRing.mem_maximalIdeal := IsLocalRing.mem_maximalIdeal


@[deprecated (since := "2024-11-11")]
alias LocalRing.not_mem_maximalIdeal := IsLocalRing.not_mem_maximalIdeal


@[deprecated (since := "2024-11-11")]
alias LocalRing.isField_iff_maximalIdeal_eq := IsLocalRing.isField_iff_maximalIdeal_eq


theorem maximalIdeal_le_jacobson (I : Ideal R) :
    IsLocalRing.maximalIdeal R ≤ I.jacobson :=
  le_sInf fun _ ⟨_, h⟩ => le_of_eq (IsLocalRing.eq_maximalIdeal h).symm


theorem jacobson_eq_maximalIdeal (I : Ideal R) (h : I ≠ ⊤) :
    I.jacobson = IsLocalRing.maximalIdeal R :=
  le_antisymm (sInf_le ⟨le_maximalIdeal h, maximalIdeal.isMaximal R⟩)
              (maximalIdeal_le_jacobson I)


@[deprecated (since := "2024-11-11")]
alias LocalRing.maximalIdeal_le_jacobson := IsLocalRing.maximalIdeal_le_jacobson


@[deprecated (since := "2024-11-11")]
alias LocalRing.jacobson_eq_maximalIdeal := IsLocalRing.jacobson_eq_maximalIdeal


theorem ker_eq_maximalIdeal [Field K] (φ : R →+* K) (hφ : Function.Surjective φ) :
    RingHom.ker φ = maximalIdeal R :=
  IsLocalRing.eq_maximalIdeal <| (RingHom.ker_isMaximal_of_surjective φ) hφ


theorem IsLocalRing.maximalIdeal_eq_bot {R : Type*} [Field R] : IsLocalRing.maximalIdeal R = ⊥ :=
  IsLocalRing.isField_iff_maximalIdeal_eq.mp (Field.toIsField R)


@[deprecated (since := "2024-11-09")]
alias LocalRing.ker_eq_maximalIdeal := IsLocalRing.ker_eq_maximalIdeal


@[deprecated (since := "2024-11-09")]
alias LocalRing.maximalIdeal_eq_bot := IsLocalRing.maximalIdeal_eq_bot


theorem IsLocalRing.of_nilradical_isMaximal [h : (nilradical R).IsMaximal] :
    IsLocalRing R := by
  /-
    R : Type u_4
    inst✝ : CommSemiring R
    h : (nilradical R).IsMaximal
    ⊢ IsLocalRing R
  -/
  refine IsLocalRing.of_unique_max_ideal ⟨nilradical R, h, fun I hI ↦ ?_⟩
  /-
    R : Type u_4
    inst✝ : CommSemiring R
    h : (nilradical R).IsMaximal
    I : Ideal R
    hI : (fun I => I.IsMaximal) I
    ⊢ Eq I (nilradical R)
  -/
  rw [nilradical_eq_sInf] at h ⊢
  /-
    R : Type u_4
    inst✝ : CommSemiring R
    h : (InfSet.sInf (setOf fun J => J.IsPrime)).IsMaximal
    I : Ideal R
    hI : I.IsMaximal
    ⊢ Eq I (InfSet.sInf (setOf fun J => J.IsPrime))
  -/
  exact (IsMaximal.eq_of_le h hI.ne_top (sInf_le hI.isPrime)).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-09")]
alias LocalRing.of_nilradical_isMaximal := IsLocalRing.of_nilradical_isMaximal


/--
Let `S` be the localization of a commutative semiring `R` at a submonoid `M` that does not
contain 0. If the nilradical of `R` is maximal then there is a `R`-algebra isomorphism between
`R` and `S`. -/
noncomputable def localizationEquivSelfOfNilradicalIsMaximal [h : (nilradical R).IsMaximal]
    (h' : (0 : R) ∉ M) [IsLocalization M S] : R ≃ₐ[R] S := by
  have (m) (hm : m ∈ M) : IsUnit m := by
    haveI := IsLocalRing.of_nilradical_isMaximal (h := h)
    apply IsLocalRing.not_mem_maximalIdeal.mp
    rw [← IsLocalRing.eq_maximalIdeal h]
    rintro ⟨k, hk⟩
    rw [← hk] at h'
    exact h' (Submonoid.pow_mem M hm k)
  /-
    R✝ : Type u_1
    S✝ : Type u_2
    K : Type u_3
    R : Type u_4
    inst✝³ : CommSemiring R
    S : Type u_5
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    h : (nilradical R).IsMaximal
    h' : Not (Membership.mem M 0)
    inst✝ : IsLocalization M S
    this : ∀ (m : R), Membership.mem M m → IsUnit m
    ⊢ AlgEquiv R R S
  -/
  exact IsLocalization.atUnits _ _ this
  /-
    🎉 no goals
  -/


