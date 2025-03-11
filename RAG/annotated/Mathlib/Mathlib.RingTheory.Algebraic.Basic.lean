@[nontriviality]
theorem is_transcendental_of_subsingleton [Subsingleton R] (x : A) : Transcendental R x :=
  fun ⟨p, h, _⟩ => h <| Subsingleton.elim p 0


theorem IsAlgebraic.nontrivial {a : A} (h : IsAlgebraic R a) : Nontrivial R := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : IsAlgebraic R a
    ⊢ Nontrivial R
  -/
  contrapose! h
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : Not (Nontrivial R)
    ⊢ Not (IsAlgebraic R a)
  -/
  rw [not_nontrivial_iff_subsingleton] at h
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    h : Subsingleton R
    ⊢ Not (IsAlgebraic R a)
  -/
  apply is_transcendental_of_subsingleton
  /-
    🎉 no goals
  -/


theorem Algebra.IsAlgebraic.nontrivial [alg : Algebra.IsAlgebraic R A] : Nontrivial R :=
  (alg.1 0).nontrivial


instance (priority := low) Algebra.transcendental_of_subsingleton [Subsingleton R] :
    Algebra.Transcendental R A :=
  ⟨⟨0, is_transcendental_of_subsingleton R 0⟩⟩


theorem Polynomial.transcendental_X : Transcendental R (X (R := R)) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Transcendental R Polynomial.X
  -/
  simp [transcendental_iff]
  /-
    🎉 no goals
  -/


theorem IsAlgebraic.of_aeval {r : A} (f : R[X]) (hf : f.natDegree ≠ 0)
    (hf' : f.leadingCoeff ∈ nonZeroDivisors R) (H : IsAlgebraic R (aeval r f)) :
    IsAlgebraic R r := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    hf : Ne f.natDegree 0
    hf' : Membership.mem (nonZeroDivisors R) f.leadingCoeff
    H : IsAlgebraic R ((Polynomial.aeval r) f)
    ⊢ IsAlgebraic R r
  -/
  obtain ⟨p, h1, h2⟩ := H
  have : (p.comp f).coeff (p.natDegree * f.natDegree) ≠ 0 := fun h ↦ h1 <| by
    rwa [coeff_comp_degree_mul_degree hf,
      mul_right_mem_nonZeroDivisors_eq_zero_iff (pow_mem hf' _),
      leadingCoeff_eq_zero] at h
  /-
    case intro.intro
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    hf : Ne f.natDegree 0
    hf' : Membership.mem (nonZeroDivisors R) f.leadingCoeff
    p : Polynomial R
    h1 : Ne p 0
    h2 : Eq ((Polynomial.aeval ((Polynomial.aeval r) f)) p) 0
    this : Ne ((p.comp f).coeff (HMul.hMul p.natDegree f.natDegree)) 0
    ⊢ IsAlgebraic R r
  -/
  exact ⟨p.comp f, fun h ↦ this (by simp [h]), by rwa [aeval_comp]⟩
  /-
    🎉 no goals
  -/


theorem Transcendental.aeval {r : A} (H : Transcendental R r) (f : R[X]) (hf : f.natDegree ≠ 0)
    (hf' : f.leadingCoeff ∈ nonZeroDivisors R) :
    Transcendental R (aeval r f) := fun h ↦ H (h.of_aeval f hf hf')


/-- If `r : A` and `f : R[X]` are transcendental over `R`, then `Polynomial.aeval r f` is also
transcendental over `R`. For the converse, see `Transcendental.of_aeval` and
`transcendental_aeval_iff`. -/
theorem Transcendental.aeval_of_transcendental {r : A} (H : Transcendental R r)
    {f : R[X]} (hf : Transcendental R f) : Transcendental R (Polynomial.aeval r f) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    H : Transcendental R r
    f : Polynomial R
    hf : Transcendental R f
    ⊢ Transcendental R ((Polynomial.aeval r) f)
  -/
  rw [transcendental_iff] at H hf ⊢
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    H : ∀ (p : Polynomial R), Eq ((Polynomial.aeval r) p) 0 → Eq p 0
    f : Polynomial R
    hf : ∀ (p : Polynomial R), Eq ((Polynomial.aeval f) p) 0 → Eq p 0
    ⊢ ∀ (p : Polynomial R), Eq ((Polynomial.aeval ((Polynomial.aeval r) f)) p) 0 → …
  -/
  intro p hp
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    H : ∀ (p : Polynomial R), Eq ((Polynomial.aeval r) p) 0 → Eq p 0
    f : Polynomial R
    hf : ∀ (p : Polynomial R), Eq ((Polynomial.aeval f) p) 0 → Eq p 0
    p : Polynomial R
    hp : Eq ((Polynomial.aeval ((Polynomial.aeval r) f)) p) 0
    ⊢ Eq p 0
  -/
  exact hf _ (H _ (by rwa [← aeval_comp, comp_eq_aeval] at hp))
  /-
    🎉 no goals
  -/


/-- If `Polynomial.aeval r f` is transcendental over `R`, then `f : R[X]` is also
transcendental over `R`. In fact, the `r` is also transcendental over `R` provided that `R`
is a field (see `transcendental_aeval_iff`). -/
theorem Transcendental.of_aeval {r : A} {f : R[X]}
    (H : Transcendental R (Polynomial.aeval r f)) : Transcendental R f := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    H : Transcendental R ((Polynomial.aeval r) f)
    ⊢ Transcendental R f
  -/
  rw [transcendental_iff] at H ⊢
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    H : ∀ (p : Polynomial R), Eq ((Polynomial.aeval ((Polynomial.aeval r) f)) p) 0 …
    ⊢ ∀ (p : Polynomial R), Eq ((Polynomial.aeval f) p) 0 → Eq p 0
  -/
  intro p hp
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    H : ∀ (p : Polynomial R), Eq ((Polynomial.aeval ((Polynomial.aeval r) f)) p) 0 …
    p : Polynomial R
    hp : Eq ((Polynomial.aeval f) p) 0
    ⊢ Eq p 0
  -/
  exact H p (by rw [← aeval_comp, comp_eq_aeval, hp, map_zero])
  /-
    🎉 no goals
  -/


theorem IsAlgebraic.of_aeval_of_transcendental {r : A} {f : R[X]}
    (H : IsAlgebraic R (aeval r f)) (hf : Transcendental R f) : IsAlgebraic R r := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    H : IsAlgebraic R ((Polynomial.aeval r) f)
    hf : Transcendental R f
    ⊢ IsAlgebraic R r
  -/
  contrapose H
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : A
    f : Polynomial R
    hf : Transcendental R f
    H : Not (IsAlgebraic R r)
    ⊢ Not (IsAlgebraic R ((Polynomial.aeval r) f))
  -/
  exact Transcendental.aeval_of_transcendental H hf
  /-
    🎉 no goals
  -/


theorem Polynomial.transcendental (f : R[X]) (hf : f.natDegree ≠ 0)
    (hf' : f.leadingCoeff ∈ nonZeroDivisors R) :
    Transcendental R f := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    hf : Ne f.natDegree 0
    hf' : Membership.mem (nonZeroDivisors R) f.leadingCoeff
    ⊢ Transcendental R f
  -/
  simpa using (transcendental_X R).aeval f hf hf'
  /-
    🎉 no goals
  -/


theorem isAlgebraic_iff_not_injective {x : A} :
    IsAlgebraic R x ↔ ¬Function.Injective (Polynomial.aeval x : R[X] →ₐ[R] A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (IsAlgebraic R x) (Not (Function.Injective ⇑(Polynomial.aeval x)))
  -/
  simp only [IsAlgebraic, injective_iff_map_eq_zero, not_forall, and_comm, exists_prop]
  /-
    🎉 no goals
  -/


/-- An element `x` is transcendental over `R` if and only if the map `Polynomial.aeval x`
is injective. This is similar to `algebraicIndependent_iff_injective_aeval`. -/
theorem transcendental_iff_injective {x : A} :
    Transcendental R x ↔ Function.Injective (Polynomial.aeval x : R[X] →ₐ[R] A) :=
  isAlgebraic_iff_not_injective.not_left


/-- An element `x` is transcendental over `R` if and only if the kernel of the ring homomorphism
`Polynomial.aeval x` is the zero ideal. This is similar to `algebraicIndependent_iff_ker_eq_bot`. -/
theorem transcendental_iff_ker_eq_bot {x : A} :
    Transcendental R x ↔ RingHom.ker (aeval (R := R) x) = ⊥ := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (Transcendental R x) (Eq (RingHom.ker (Polynomial.aeval x)) Bot.bot)
  -/
  rw [transcendental_iff_injective, RingHom.injective_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


theorem Algebra.isAlgebraic_of_not_injective (h : ¬ Function.Injective (algebraMap R A)) :
    Algebra.IsAlgebraic R A where
  isAlgebraic a := isAlgebraic_iff_not_injective.mpr
                      /-
                        R : Type u
                        A : Type v
                        inst✝² : CommRing R
                        inst✝¹ : Ring A
                        inst✝ : Algebra R A
                        h : Not (Function.Injective ⇑(algebraMap R A))
                        a : A
                        inj : Function.Injective ⇑(Polynomial.aeval a)
                        ⊢ Function.Injective ⇑(algebraMap R A)
                      -/
    fun inj ↦ h <| by convert inj.comp C_injective; ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem Algebra.injective_of_transcendental [h : Algebra.Transcendental R A] :
    Function.Injective (algebraMap R A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h : Algebra.Transcendental R A
    ⊢ Function.Injective ⇑(algebraMap R A)
  -/
  rw [transcendental_iff_not_isAlgebraic] at h
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h : Not (Algebra.IsAlgebraic R A)
    ⊢ Function.Injective ⇑(algebraMap R A)
  -/
  contrapose! h
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    h : Not (Function.Injective ⇑(algebraMap R A))
    ⊢ Algebra.IsAlgebraic R A
  -/
  exact isAlgebraic_of_not_injective h
  /-
    🎉 no goals
  -/


theorem isAlgebraic_zero [Nontrivial R] : IsAlgebraic R (0 : A) :=
  ⟨_, X_ne_zero, aeval_X 0⟩


/-- An element of `R` is algebraic, when viewed as an element of the `R`-algebra `A`. -/
theorem isAlgebraic_algebraMap [Nontrivial R] (x : R) : IsAlgebraic R (algebraMap R A x) :=
                            /-
                              R : Type u
                              A : Type v
                              inst✝³ : CommRing R
                              inst✝² : Ring A
                              inst✝¹ : Algebra R A
                              inst✝ : Nontrivial R
                              x : R
                              ⊢ Eq ((Polynomial.aeval ((algebraMap R A) x)) (HSub.hSub Polynomial.X (Polynom …
                            -/
  ⟨_, X_sub_C_ne_zero x, by rw [map_sub, aeval_X, aeval_C, sub_self]⟩
                            /-
                              🎉 no goals
                            -/


theorem isAlgebraic_one [Nontrivial R] : IsAlgebraic R (1 : A) := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    ⊢ IsAlgebraic R 1
  -/
  rw [← map_one (algebraMap R A)]
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    ⊢ IsAlgebraic R ((algebraMap R A) 1)
  -/
  exact isAlgebraic_algebraMap 1
  /-
    🎉 no goals
  -/


theorem isAlgebraic_nat [Nontrivial R] (n : ℕ) : IsAlgebraic R (n : A) := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    n : Nat
    ⊢ IsAlgebraic R ↑n
  -/
  rw [← map_natCast (_ : R →+* A) n]
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    n : Nat
    ⊢ IsAlgebraic R (?m.39204 ↑n)
  -/
  exact isAlgebraic_algebraMap (Nat.cast n)
  /-
    🎉 no goals
  -/


theorem isAlgebraic_int [Nontrivial R] (n : ℤ) : IsAlgebraic R (n : A) := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    n : Int
    ⊢ IsAlgebraic R ↑n
  -/
  rw [← map_intCast (algebraMap R A)]
  /-
    R : Type u
    A : Type v
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    n : Int
    ⊢ IsAlgebraic R ((algebraMap R A) ↑n)
  -/
  exact isAlgebraic_algebraMap (Int.cast n)
  /-
    🎉 no goals
  -/


theorem isAlgebraic_rat (R : Type u) {A : Type v} [DivisionRing A] [Field R] [Algebra R A] (n : ℚ) :
    IsAlgebraic R (n : A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : DivisionRing A
    inst✝¹ : Field R
    inst✝ : Algebra R A
    n : Rat
    ⊢ IsAlgebraic R ↑n
  -/
  rw [← map_ratCast (algebraMap R A)]
  /-
    R : Type u
    A : Type v
    inst✝² : DivisionRing A
    inst✝¹ : Field R
    inst✝ : Algebra R A
    n : Rat
    ⊢ IsAlgebraic R ((algebraMap R A) ↑n)
  -/
  exact isAlgebraic_algebraMap (Rat.cast n)
  /-
    🎉 no goals
  -/


theorem isAlgebraic_of_mem_rootSet {R : Type u} {A : Type v} [Field R] [Field A] [Algebra R A]
    {p : R[X]} {x : A} (hx : x ∈ p.rootSet A) : IsAlgebraic R x :=
  ⟨p, ne_zero_of_mem_rootSet hx, aeval_eq_zero_of_mem_rootSet hx⟩


variable (S) in
theorem IsLocalization.isAlgebraic [Nontrivial R] (M : Submonoid R) [IsLocalization M S] :
    Algebra.IsAlgebraic R S where
  isAlgebraic x := by
    /-
      R : Type u
      S : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Nontrivial R
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : S
      ⊢ IsAlgebraic R x
    -/
    obtain rfl | hx := eq_or_ne x 0
      /-
        case inl
        R : Type u
        S : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : Nontrivial R
        M : Submonoid R
        inst✝ : IsLocalization M S
        ⊢ IsAlgebraic R 0
      -/
    · exact isAlgebraic_zero
      /-
        🎉 no goals
      -/
    /-
      case inr
      R : Type u
      S : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Nontrivial R
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : S
      hx : Ne x 0
      ⊢ IsAlgebraic R x
    -/
    have ⟨⟨r, m⟩, h⟩ := surj M x
    /-
      case inr
      R : Type u
      S : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Nontrivial R
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : S
      hx : Ne x 0
      r : R
      m : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMap …
      ⊢ IsAlgebraic R x
    -/
    refine ⟨C m.1 * X - C r, fun eq ↦ hx ?_, by simpa [sub_eq_zero, mul_comm x] using h⟩
    /-
      case inr
      R : Type u
      S : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Nontrivial R
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : S
      hx : Ne x 0
      r : R
      m : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMap …
      eq : Eq (HSub.hSub (HMul.hMul (Polynomial.C ↑m) Polynomial.X) (Polynomial.C r) …
      ⊢ Eq x 0
    -/
    rwa [← eq_mk'_iff_mul_eq, show r = 0 by simpa using congr(coeff $eq 0), mk'_zero] at h
    /-
      🎉 no goals
    -/


protected theorem IsAlgebraic.algebraMap {a : S} :
    IsAlgebraic R a → IsAlgebraic R (algebraMap S A a) := fun ⟨f, hf₁, hf₂⟩ =>
              /-
                R : Type u
                S : Type u_1
                A : Type v
                inst✝⁶ : CommRing R
                inst✝⁵ : CommRing S
                inst✝⁴ : Ring A
                inst✝³ : Algebra R A
                inst✝² : Algebra R S
                inst✝¹ : Algebra S A
                inst✝ : IsScalarTower R S A
                a : S
                x✝ : IsAlgebraic R a
                f : Polynomial R
                hf₁ : Ne f 0
                hf₂ : Eq ((Polynomial.aeval a) f) 0
                ⊢ Eq ((Polynomial.aeval ((algebraMap S A) a)) f) 0
              -/
  ⟨f, hf₁, by rw [aeval_algebraMap_apply, hf₂, map_zero]⟩
              /-
                🎉 no goals
              -/


/-- This is slightly more general than `IsAlgebraic.algebraMap` in that it
  allows noncommutative intermediate rings `A`. -/
protected theorem IsAlgebraic.algHom (f : A →ₐ[R] B) {a : A}
    (h : IsAlgebraic R a) : IsAlgebraic R (f a) :=
  let ⟨p, hp, ha⟩ := h
             /-
               R : Type u
               A : Type v
               inst✝⁴ : CommRing R
               inst✝³ : Ring A
               inst✝² : Algebra R A
               B : Type u_2
               inst✝¹ : Ring B
               inst✝ : Algebra R B
               f : AlgHom R A B
               a : A
               h : IsAlgebraic R a
               p : Polynomial R
               hp : Ne p 0
               ha : Eq ((Polynomial.aeval a) p) 0
               ⊢ Eq ((Polynomial.aeval (f a)) p) 0
             -/
  ⟨p, hp, by rw [aeval_algHom, f.comp_apply, ha, map_zero]⟩
             /-
               🎉 no goals
             -/


theorem isAlgebraic_algHom_iff (f : A →ₐ[R] B) (hf : Function.Injective f)
    {a : A} : IsAlgebraic R (f a) ↔ IsAlgebraic R a :=
                                        /-
                                          R : Type u
                                          A : Type v
                                          inst✝⁴ : CommRing R
                                          inst✝³ : Ring A
                                          inst✝² : Algebra R A
                                          B : Type u_2
                                          inst✝¹ : Ring B
                                          inst✝ : Algebra R B
                                          f : AlgHom R A B
                                          hf : Function.Injective ⇑f
                                          a : A
                                          x✝ : IsAlgebraic R (f a)
                                          p : Polynomial R
                                          hp0 : Ne p 0
                                          hp : Eq ((Polynomial.aeval (f a)) p) 0
                                          ⊢ Eq (f ((Polynomial.aeval a) p)) (f 0)
                                        -/
  ⟨fun ⟨p, hp0, hp⟩ ↦ ⟨p, hp0, hf <| by rwa [map_zero, ← f.comp_apply, ← aeval_algHom]⟩,
                                        /-
                                          🎉 no goals
                                        -/
    IsAlgebraic.algHom f⟩


theorem IsAlgebraic.ringHom_of_comp_eq (halg : IsAlgebraic R a)
    (hf : Function.Injective f)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    IsAlgebraic S (g a) := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    halg : IsAlgebraic R a
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ IsAlgebraic S (g a)
  -/
  obtain ⟨p, h1, h2⟩ := halg
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : Polynomial R
    h1 : Ne p 0
    h2 : Eq ((Polynomial.aeval a) p) 0
    ⊢ IsAlgebraic S (g a)
  -/
  refine ⟨p.map f, (Polynomial.map_ne_zero_iff hf).2 h1, ?_⟩
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : Polynomial R
    h1 : Ne p 0
    h2 : Eq ((Polynomial.aeval a) p) 0
    ⊢ Eq ((Polynomial.aeval (g a)) (Polynomial.map (↑f) p)) 0
  -/
  change aeval ((g : A →+* B) a) _ = 0
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : Polynomial R
    h1 : Ne p 0
    h2 : Eq ((Polynomial.aeval a) p) 0
    ⊢ Eq ((Polynomial.aeval (↑g a)) (Polynomial.map (↑f) p)) 0
  -/
  rw [← map_aeval_eq_aeval_map h, h2, map_zero]
  /-
    🎉 no goals
  -/


theorem Transcendental.of_ringHom_of_comp_eq (H : Transcendental S (g a))
    (hf : Function.Injective f)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Transcendental R a := fun halg ↦ H (halg.ringHom_of_comp_eq f g hf h)


theorem Algebra.IsAlgebraic.ringHom_of_comp_eq [Algebra.IsAlgebraic R A]
    (hf : Function.Injective f) (hg : Function.Surjective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.IsAlgebraic S B := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Ring A
    inst✝⁷ : Algebra R A
    B : Type u_2
    inst✝⁶ : Ring B
    inst✝⁵ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝⁴ : FunLike FRS R S
    inst✝³ : RingHomClass FRS R S
    inst✝² : FunLike FAB A B
    inst✝¹ : RingHomClass FAB A B
    f : FRS
    g : FAB
    inst✝ : Algebra.IsAlgebraic R A
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ Algebra.IsAlgebraic S B
  -/
  refine ⟨fun b ↦ ?_⟩
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Ring A
    inst✝⁷ : Algebra R A
    B : Type u_2
    inst✝⁶ : Ring B
    inst✝⁵ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝⁴ : FunLike FRS R S
    inst✝³ : RingHomClass FRS R S
    inst✝² : FunLike FAB A B
    inst✝¹ : RingHomClass FAB A B
    f : FRS
    g : FAB
    inst✝ : Algebra.IsAlgebraic R A
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    b : B
    ⊢ IsAlgebraic S b
  -/
  obtain ⟨a, rfl⟩ := hg b
  /-
    case intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Ring A
    inst✝⁷ : Algebra R A
    B : Type u_2
    inst✝⁶ : Ring B
    inst✝⁵ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝⁴ : FunLike FRS R S
    inst✝³ : RingHomClass FRS R S
    inst✝² : FunLike FAB A B
    inst✝¹ : RingHomClass FAB A B
    f : FRS
    g : FAB
    inst✝ : Algebra.IsAlgebraic R A
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    a : A
    ⊢ IsAlgebraic S (g a)
  -/
  exact (Algebra.IsAlgebraic.isAlgebraic a).ringHom_of_comp_eq f g hf h
  /-
    🎉 no goals
  -/


theorem Algebra.Transcendental.of_ringHom_of_comp_eq [H : Algebra.Transcendental S B]
    (hf : Function.Injective f) (hg : Function.Surjective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.Transcendental R A := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : Algebra.Transcendental S B
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ Algebra.Transcendental R A
  -/
  rw [Algebra.transcendental_iff_not_isAlgebraic] at H ⊢
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : Not (Algebra.IsAlgebraic S B)
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ Not (Algebra.IsAlgebraic R A)
  -/
  exact fun halg ↦ H (halg.ringHom_of_comp_eq f g hf hg h)
  /-
    🎉 no goals
  -/


theorem IsAlgebraic.of_ringHom_of_comp_eq (halg : IsAlgebraic S (g a))
    (hf : Function.Surjective f) (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    IsAlgebraic R a := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    halg : IsAlgebraic S (g a)
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ IsAlgebraic R a
  -/
  obtain ⟨p, h1, h2⟩ := halg
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : Polynomial S
    h1 : Ne p 0
    h2 : Eq ((Polynomial.aeval (g a)) p) 0
    ⊢ IsAlgebraic R a
  -/
  obtain ⟨q, rfl⟩ := map_surjective (f : R →+* S) hf p
  /-
    case intro.intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    q : Polynomial R
    h1 : Ne (Polynomial.map (↑f) q) 0
    h2 : Eq ((Polynomial.aeval (g a)) (Polynomial.map (↑f) q)) 0
    ⊢ IsAlgebraic R a
  -/
  refine ⟨q, fun h' ↦ by simp [h'] at h1, hg ?_⟩
  /-
    case intro.intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    q : Polynomial R
    h1 : Ne (Polynomial.map (↑f) q) 0
    h2 : Eq ((Polynomial.aeval (g a)) (Polynomial.map (↑f) q)) 0
    ⊢ Eq (g ((Polynomial.aeval a) q)) (g 0)
  -/
  change aeval ((g : A →+* B) a) _ = 0 at h2
  /-
    case intro.intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    q : Polynomial R
    h1 : Ne (Polynomial.map (↑f) q) 0
    h2 : Eq ((Polynomial.aeval (↑g a)) (Polynomial.map (↑f) q)) 0
    ⊢ Eq (g ((Polynomial.aeval a) q)) (g 0)
  -/
  change (g : A →+* B) _ = _
  /-
    case intro.intro.intro
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    a : A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    q : Polynomial R
    h1 : Ne (Polynomial.map (↑f) q) 0
    h2 : Eq ((Polynomial.aeval (↑g a)) (Polynomial.map (↑f) q)) 0
    ⊢ Eq (↑g ((Polynomial.aeval a) q)) (g 0)
  -/
  rw [map_zero, map_aeval_eq_aeval_map h, h2]
  /-
    🎉 no goals
  -/


theorem Transcendental.ringHom_of_comp_eq (H : Transcendental R a)
    (hf : Function.Surjective f) (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Transcendental S (g a) := fun halg ↦ H (halg.of_ringHom_of_comp_eq f g hf hg h)


theorem Algebra.IsAlgebraic.of_ringHom_of_comp_eq [Algebra.IsAlgebraic S B]
    (hf : Function.Surjective f) (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.IsAlgebraic R A :=
  ⟨fun a ↦ (Algebra.IsAlgebraic.isAlgebraic (g a)).of_ringHom_of_comp_eq f g hf hg h⟩


theorem Algebra.Transcendental.ringHom_of_comp_eq [H : Algebra.Transcendental R A]
    (hf : Function.Surjective f) (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.Transcendental S B := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : Algebra.Transcendental R A
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ Algebra.Transcendental S B
  -/
  rw [Algebra.transcendental_iff_not_isAlgebraic] at H ⊢
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Ring A
    inst✝⁶ : Algebra R A
    B : Type u_2
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra S B
    FRS : Type u_3
    FAB : Type u_4
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : Not (Algebra.IsAlgebraic R A)
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ Not (Algebra.IsAlgebraic S B)
  -/
  exact fun halg ↦ H (halg.of_ringHom_of_comp_eq f g hf hg h)
  /-
    🎉 no goals
  -/


theorem isAlgebraic_ringHom_iff_of_comp_eq
    (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) {a : A} :
    IsAlgebraic S (g a) ↔ IsAlgebraic R a :=
  ⟨fun H ↦ H.of_ringHom_of_comp_eq f g (EquivLike.surjective f) hg h,
    fun H ↦ H.ringHom_of_comp_eq f g (EquivLike.injective f) h⟩


theorem transcendental_ringHom_iff_of_comp_eq
    (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) {a : A} :
    Transcendental S (g a) ↔ Transcendental R a :=
  not_congr (isAlgebraic_ringHom_iff_of_comp_eq f g hg h)


theorem Algebra.isAlgebraic_ringHom_iff_of_comp_eq
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.IsAlgebraic S B ↔ Algebra.IsAlgebraic R A :=
  ⟨fun H ↦ H.of_ringHom_of_comp_eq f g (EquivLike.surjective f) (EquivLike.injective g) h,
    fun H ↦ H.ringHom_of_comp_eq f g (EquivLike.injective f) (EquivLike.surjective g) h⟩


theorem Algebra.transcendental_ringHom_iff_of_comp_eq
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    Algebra.Transcendental S B ↔ Algebra.Transcendental R A := by
  simp_rw [Algebra.transcendental_iff_not_isAlgebraic,
    Algebra.isAlgebraic_ringHom_iff_of_comp_eq f g h]


theorem Algebra.IsAlgebraic.of_injective (f : A →ₐ[R] B) (hf : Function.Injective f)
    [Algebra.IsAlgebraic R B] : Algebra.IsAlgebraic R A :=
  ⟨fun _ ↦ (isAlgebraic_algHom_iff f hf).mp (Algebra.IsAlgebraic.isAlgebraic _)⟩


/-- Transfer `Algebra.IsAlgebraic` across an `AlgEquiv`. -/
theorem AlgEquiv.isAlgebraic (e : A ≃ₐ[R] B)
    [Algebra.IsAlgebraic R A] : Algebra.IsAlgebraic R B :=
  Algebra.IsAlgebraic.of_injective e.symm.toAlgHom e.symm.injective


theorem AlgEquiv.isAlgebraic_iff (e : A ≃ₐ[R] B) :
    Algebra.IsAlgebraic R A ↔ Algebra.IsAlgebraic R B :=
  ⟨fun _ ↦ e.isAlgebraic, fun _ ↦ e.symm.isAlgebraic⟩


theorem isAlgebraic_algebraMap_iff {a : S} (h : Function.Injective (algebraMap S A)) :
    IsAlgebraic R (algebraMap S A a) ↔ IsAlgebraic R a :=
  isAlgebraic_algHom_iff (IsScalarTower.toAlgHom R S A) h


theorem transcendental_algebraMap_iff {a : S} (h : Function.Injective (algebraMap S A)) :
    Transcendental R (algebraMap S A a) ↔ Transcendental R a := by
  /-
    R : Type u
    S : Type u_1
    A : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : Algebra R S
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    a : S
    h : Function.Injective ⇑(algebraMap S A)
    ⊢ Iff (Transcendental R ((algebraMap S A) a)) (Transcendental R a)
  -/
  simp_rw [Transcendental, isAlgebraic_algebraMap_iff h]
  /-
    🎉 no goals
  -/


theorem isAlgebraic_iff_isAlgebraic_val {S : Subalgebra R A} {x : S} :
    _root_.IsAlgebraic R x ↔ _root_.IsAlgebraic R x.1 :=
  (isAlgebraic_algHom_iff S.val Subtype.val_injective).symm


theorem isAlgebraic_of_isAlgebraic_bot {x : S} (halg : _root_.IsAlgebraic (⊥ : Subalgebra R S) x) :
    _root_.IsAlgebraic R x :=
  halg.of_ringHom_of_comp_eq (algebraMap R (⊥ : Subalgebra R S))
                       /-
                         R : Type u
                         S : Type u_1
                         inst✝² : CommRing R
                         inst✝¹ : CommRing S
                         inst✝ : Algebra R S
                         x : S
                         halg : _root_.IsAlgebraic (Subtype fun x => Membership.mem Bot.bot x) x
                         ⊢ Function.Surjective ⇑(algebraMap R (Subtype fun x => Membership.mem Bot.bot  …
                       -/
    (RingHom.id S) (by rintro ⟨_, r, rfl⟩; exact ⟨r, rfl⟩) Function.injective_id rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem isAlgebraic_bot_iff (h : Function.Injective (algebraMap R S)) {x : S} :
    _root_.IsAlgebraic (⊥ : Subalgebra R S) x ↔ _root_.IsAlgebraic R x :=
  isAlgebraic_ringHom_iff_of_comp_eq (Algebra.botEquivOfInjective h).symm (RingHom.id S)
                              /-
                                R : Type u
                                S : Type u_1
                                inst✝² : CommRing R
                                inst✝¹ : CommRing S
                                inst✝ : Algebra R S
                                h : Function.Injective ⇑(algebraMap R S)
                                x : S
                                ⊢ Eq ((algebraMap (Subtype fun x => Membership.mem Bot.bot x) S).comp ↑(Algebr …
                              -/
    Function.injective_id (by rfl)
                              /-
                                🎉 no goals
                              -/


variable (R S) in
theorem algebra_isAlgebraic_of_algebra_isAlgebraic_bot_left
    [Algebra.IsAlgebraic (⊥ : Subalgebra R S) S] : Algebra.IsAlgebraic R S :=
  Algebra.IsAlgebraic.of_ringHom_of_comp_eq (algebraMap R (⊥ : Subalgebra R S))
                       /-
                         R : Type u
                         S : Type u_1
                         inst✝³ : CommRing R
                         inst✝² : CommRing S
                         inst✝¹ : Algebra R S
                         inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem Bot.bot x) S
                         ⊢ Function.Surjective ⇑(algebraMap R (Subtype fun x => Membership.mem Bot.bot  …
                       -/
                                           /-
                                             🎉 no goals
                                           -/
    (RingHom.id S) (by rintro ⟨_, r, rfl⟩; exact ⟨r, rfl⟩) Function.injective_id (by ext; rfl)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem algebra_isAlgebraic_bot_left_iff (h : Function.Injective (algebraMap R S)) :
    Algebra.IsAlgebraic (⊥ : Subalgebra R S) S ↔ Algebra.IsAlgebraic R S := by
  /-
    R : Type u
    S : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    h : Function.Injective ⇑(algebraMap R S)
    ⊢ Iff (Algebra.IsAlgebraic (Subtype fun x => Membership.mem Bot.bot x) S) (Alg …
  -/
  simp_rw [Algebra.isAlgebraic_def, isAlgebraic_bot_iff h]
  /-
    🎉 no goals
  -/


instance algebra_isAlgebraic_bot_right [Nontrivial R] :
    Algebra.IsAlgebraic R (⊥ : Subalgebra R S) :=
      /-
        R : Type u
        S : Type u_1
        A : Type v
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Ring A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R S
        inst✝² : Algebra S A
        inst✝¹ : IsScalarTower R S A
        inst✝ : Nontrivial R
        ⊢ ∀ (x : Subtype fun x => Membership.mem Bot.bot x), _root_.IsAlgebraic R x
      -/
  ⟨by rintro ⟨_, x, rfl⟩; exact isAlgebraic_algebraMap _⟩
                          /-
                            🎉 no goals
                          -/


theorem IsAlgebraic.of_pow {r : A} {n : ℕ} (hn : 0 < n) (ht : IsAlgebraic R (r ^ n)) :
    IsAlgebraic R r :=
  have ⟨p, p_nonzero, hp⟩ := ht
         /-
           R : Type u
           A : Type v
           inst✝² : CommRing R
           inst✝¹ : Ring A
           inst✝ : Algebra R A
           r : A
           n : Nat
           hn : LT.lt 0 n
           ht : IsAlgebraic R (HPow.hPow r n)
           p : Polynomial R
           p_nonzero : Ne p 0
           hp : Eq ((Polynomial.aeval (HPow.hPow r n)) p) 0
           ⊢ Ne (?m.173920 p p_nonzero hp) 0
         -/
         /-
           🎉 no goals
         -/
  ⟨_, by rwa [expand_ne_zero hn], by rwa [expand_aeval n p r]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem Transcendental.pow {r : A} (ht : Transcendental R r) {n : ℕ} (hn : 0 < n) :
    Transcendental R (r ^ n) := fun ht' ↦ ht <| ht'.of_pow hn


lemma IsAlgebraic.invOf {x : S} [Invertible x] (h : IsAlgebraic R x) : IsAlgebraic R (⅟ x) := by
  /-
    R : Type u
    S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    x : S
    inst✝ : Invertible x
    h : IsAlgebraic R x
    ⊢ IsAlgebraic R (Invertible.invOf x)
  -/
  obtain ⟨p, hp, hp'⟩ := h
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    x : S
    inst✝ : Invertible x
    p : Polynomial R
    hp : Ne p 0
    hp' : Eq ((Polynomial.aeval x) p) 0
    ⊢ IsAlgebraic R (Invertible.invOf x)
  -/
  refine ⟨p.reverse, by simpa using hp, ?_⟩
  /-
    case intro.intro
    R : Type u
    S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    x : S
    inst✝ : Invertible x
    p : Polynomial R
    hp : Ne p 0
    hp' : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq ((Polynomial.aeval (Invertible.invOf x)) p.reverse) 0
  -/
  rwa [Polynomial.aeval_def, Polynomial.eval₂_reverse_eq_zero_iff, ← Polynomial.aeval_def]
  /-
    🎉 no goals
  -/


lemma IsAlgebraic.invOf_iff {x : S} [Invertible x] :
    IsAlgebraic R (⅟ x) ↔ IsAlgebraic R x :=
  ⟨IsAlgebraic.invOf, IsAlgebraic.invOf⟩


lemma IsAlgebraic.inv_iff {K} [Field K] [Algebra R K] {x : K} :
    IsAlgebraic R (x⁻¹) ↔ IsAlgebraic R x := by
  /-
    R : Type u
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    x : K
    ⊢ Iff (IsAlgebraic R (Inv.inv x)) (IsAlgebraic R x)
  -/
  by_cases hx : x = 0
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      x : K
      hx : Eq x 0
      ⊢ Iff (IsAlgebraic R (Inv.inv x)) (IsAlgebraic R x)
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    x : K
    hx : Not (Eq x 0)
    ⊢ Iff (IsAlgebraic R (Inv.inv x)) (IsAlgebraic R x)
  -/
  letI := invertibleOfNonzero hx
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    x : K
    hx : Not (Eq x 0)
    this : Invertible x := invertibleOfNonzero hx
    ⊢ Iff (IsAlgebraic R (Inv.inv x)) (IsAlgebraic R x)
  -/
  exact IsAlgebraic.invOf_iff (R := R) (x := x)
  /-
    🎉 no goals
  -/


alias ⟨_, IsAlgebraic.inv⟩ := IsAlgebraic.inv_iff


/-- If `x` is algebraic over `R`, then `x` is algebraic over `S` when `S` is an extension of `R`,
  and the map from `R` to `S` is injective. -/
theorem IsAlgebraic.extendScalars (hinj : Function.Injective (algebraMap R S)) {x : A}
    (A_alg : IsAlgebraic R x) : IsAlgebraic S x :=
  let ⟨p, hp₁, hp₂⟩ := A_alg
  ⟨p.map (algebraMap _ _), by
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Ring A
      inst✝³ : Algebra R S
      inst✝² : Algebra S A
      inst✝¹ : Algebra R A
      inst✝ : IsScalarTower R S A
      hinj : Function.Injective ⇑(algebraMap R S)
      x : A
      A_alg : IsAlgebraic R x
      p : Polynomial R
      hp₁ : Ne p 0
      hp₂ : Eq ((Polynomial.aeval x) p) 0
      ⊢ Ne (Polynomial.map (algebraMap R S) p) 0
    -/
    /-
      🎉 no goals
    -/
    rwa [Ne, ← degree_eq_bot, degree_map_eq_of_injective hinj, degree_eq_bot], by simpa⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[deprecated (since := "2024-11-18")]
alias IsAlgebraic.tower_top_of_injective := IsAlgebraic.extendScalars


/-- A special case of `IsAlgebraic.extendScalars`. This is extracted as a theorem
  because in some cases `IsAlgebraic.extendScalars` will just runs out of memory. -/
theorem IsAlgebraic.tower_top_of_subalgebra_le
    {A B : Subalgebra R S} (hle : A ≤ B) {x : S}
    (h : IsAlgebraic A x) : IsAlgebraic B x := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    hle : LE.le A B
    x : S
    h : IsAlgebraic (Subtype fun x => Membership.mem A x) x
    ⊢ IsAlgebraic (Subtype fun x => Membership.mem B x) x
  -/
  letI : Algebra A B := (Subalgebra.inclusion hle).toAlgebra
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    hle : LE.le A B
    x : S
    h : IsAlgebraic (Subtype fun x => Membership.mem A x) x
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ IsAlgebraic (Subtype fun x => Membership.mem B x) x
  -/
  haveI : IsScalarTower A B S := .of_algebraMap_eq fun _ ↦ rfl
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    hle : LE.le A B
    x : S
    h : IsAlgebraic (Subtype fun x => Membership.mem A x) x
    this✝ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Member …
    this : IsScalarTower (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    ⊢ IsAlgebraic (Subtype fun x => Membership.mem B x) x
  -/
  exact h.extendScalars (Subalgebra.inclusion_injective hle)
  /-
    🎉 no goals
  -/


/-- If `x` is transcendental over `S`, then `x` is transcendental over `R` when `S` is an extension
  of `R`, and the map from `R` to `S` is injective. -/
theorem Transcendental.restrictScalars (hinj : Function.Injective (algebraMap R S)) {x : A}
    (h : Transcendental S x) : Transcendental R x := fun H ↦ h (H.extendScalars hinj)


@[deprecated (since := "2024-11-18")]
alias Transcendental.of_tower_top_of_injective := Transcendental.restrictScalars


/-- A special case of `Transcendental.restrictScalars`. This is extracted as a theorem
  because in some cases `Transcendental.restrictScalars` will just runs out of memory. -/
theorem Transcendental.of_tower_top_of_subalgebra_le
    {A B : Subalgebra R S} (hle : A ≤ B) {x : S}
    (h : Transcendental B x) : Transcendental A x :=
  fun H ↦ h (H.tower_top_of_subalgebra_le hle)


/-- If A is an algebraic algebra over R, then A is algebraic over S when S is an extension of R,
  and the map from `R` to `S` is injective. -/
theorem Algebra.IsAlgebraic.extendScalars (hinj : Function.Injective (algebraMap R S))
    [Algebra.IsAlgebraic R A] : Algebra.IsAlgebraic S A :=
  ⟨fun _ ↦ _root_.IsAlgebraic.extendScalars hinj (Algebra.IsAlgebraic.isAlgebraic _)⟩


@[deprecated (since := "2024-11-18")]
alias Algebra.IsAlgebraic.tower_top_of_injective := Algebra.IsAlgebraic.extendScalars


theorem Algebra.IsAlgebraic.tower_bot_of_injective [Algebra.IsAlgebraic R A]
    (hinj : Function.Injective (algebraMap S A)) :
    Algebra.IsAlgebraic R S where
  isAlgebraic x := by
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra S A
      inst✝² : Algebra R A
      inst✝¹ : IsScalarTower R S A
      inst✝ : Algebra.IsAlgebraic R A
      hinj : Function.Injective ⇑(algebraMap S A)
      x : S
      ⊢ IsAlgebraic R x
    -/
    simpa [isAlgebraic_algebraMap_iff hinj] using isAlgebraic (R := R) (A := A) (algebraMap _ _ x)
    /-
      🎉 no goals
    -/


/-- If `x` is algebraic over `K`, then `x` is algebraic over `L` when `L` is an extension of `K` -/
@[stacks 09GF "part one"]
theorem IsAlgebraic.tower_top {x : A} (A_alg : IsAlgebraic K x) :
    IsAlgebraic L x :=
  A_alg.extendScalars (algebraMap K L).injective


variable {L} (K) in
/-- If `x` is transcendental over `L`, then `x` is transcendental over `K` when
  `L` is an extension of `K` -/
theorem Transcendental.of_tower_top {x : A} (h : Transcendental L x) :
    Transcendental K x := fun H ↦ h (H.tower_top L)


/-- If A is an algebraic algebra over K, then A is algebraic over L when L is an extension of K -/
@[stacks 09GF "part two"]
theorem Algebra.IsAlgebraic.tower_top [Algebra.IsAlgebraic K A] : Algebra.IsAlgebraic L A :=
  Algebra.IsAlgebraic.extendScalars (algebraMap K L).injective


theorem Algebra.IsAlgebraic.tower_bot (K L A : Type*) [CommRing K] [Field L] [Ring A]
    [Algebra K L] [Algebra L A] [Algebra K A] [IsScalarTower K L A]
    [Nontrivial A] [Algebra.IsAlgebraic K A] :
    Algebra.IsAlgebraic K L :=
  tower_bot_of_injective (algebraMap L A).injective


theorem algHom_bijective [NoZeroSMulDivisors K L] [Algebra.IsAlgebraic K L] (f : L →ₐ[K] L) :
    Function.Bijective f := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : NoZeroSMulDivisors K L
    inst✝ : Algebra.IsAlgebraic K L
    f : AlgHom K L L
    ⊢ Function.Bijective ⇑f
  -/
  refine ⟨f.injective, fun b ↦ ?_⟩
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : NoZeroSMulDivisors K L
    inst✝ : Algebra.IsAlgebraic K L
    f : AlgHom K L L
    b : L
    ⊢ Exists fun a => Eq (f a) b
  -/
  obtain ⟨p, hp, he⟩ := Algebra.IsAlgebraic.isAlgebraic (R := K) b
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : NoZeroSMulDivisors K L
    inst✝ : Algebra.IsAlgebraic K L
    f : AlgHom K L L
    b : L
    p : Polynomial K
    hp : Ne p 0
    he : Eq ((Polynomial.aeval b) p) 0
    ⊢ Exists fun a => Eq (f a) b
  -/
  let f' : p.rootSet L → p.rootSet L := (rootSet_maps_to' (fun x ↦ x) f).restrict f _ _
  have : f'.Surjective := Finite.injective_iff_surjective.1
    fun _ _ h ↦ Subtype.eq <| f.injective <| Subtype.ext_iff.1 h
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : NoZeroSMulDivisors K L
    inst✝ : Algebra.IsAlgebraic K L
    f : AlgHom K L L
    b : L
    p : Polynomial K
    hp : Ne p 0
    he : Eq ((Polynomial.aeval b) p) 0
    f' : ↑(p.rootSet L) → ↑(p.rootSet L) := Set.MapsTo.restrict (⇑f) (p.rootSet L) …
    this : Function.Surjective f'
    ⊢ Exists fun a => Eq (f a) b
  -/
  obtain ⟨a, ha⟩ := this ⟨b, mem_rootSet.2 ⟨hp, he⟩⟩
  /-
    case intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : NoZeroSMulDivisors K L
    inst✝ : Algebra.IsAlgebraic K L
    f : AlgHom K L L
    b : L
    p : Polynomial K
    hp : Ne p 0
    he : Eq ((Polynomial.aeval b) p) 0
    f' : ↑(p.rootSet L) → ↑(p.rootSet L) := Set.MapsTo.restrict (⇑f) (p.rootSet L) …
    this : Function.Surjective f'
    a : ↑(p.rootSet L)
    ha : Eq (f' a) ⟨b, ⋯⟩
    ⊢ Exists fun a => Eq (f a) b
  -/
  exact ⟨a, Subtype.ext_iff.1 ha⟩
  /-
    🎉 no goals
  -/


theorem algHom_bijective₂ [NoZeroSMulDivisors K L] [Field R] [Algebra K R]
    [Algebra.IsAlgebraic K L] (f : L →ₐ[K] R) (g : R →ₐ[K] L) :
    Function.Bijective f ∧ Function.Bijective g :=
  (g.injective.bijective₂_of_surjective f.injective (algHom_bijective <| g.comp f).2).symm


theorem bijective_of_isScalarTower [NoZeroSMulDivisors K L] [Algebra.IsAlgebraic K L]
    [Field R] [Algebra K R] [Algebra L R] [IsScalarTower K L R] (f : R →ₐ[K] L) :
    Function.Bijective f :=
  (algHom_bijective₂ (IsScalarTower.toAlgHom K L R) f).2


theorem bijective_of_isScalarTower' [Field R] [Algebra K R]
    [NoZeroSMulDivisors K R]
    [Algebra.IsAlgebraic K R] [Algebra L R] [IsScalarTower K L R] (f : R →ₐ[K] L) :
    Function.Bijective f :=
  (algHom_bijective₂ f (IsScalarTower.toAlgHom K L R)).1


/-- Bijection between algebra equivalences and algebra homomorphisms -/
@[simps]
noncomputable def algEquivEquivAlgHom [NoZeroSMulDivisors K L] [Algebra.IsAlgebraic K L] :
    (L ≃ₐ[K] L) ≃* (L →ₐ[K] L) where
  toFun ϕ := ϕ.toAlgHom
  invFun ϕ := AlgEquiv.ofBijective ϕ (algHom_bijective ϕ)
                   /-
                     K : Type u_1
                     L : Type u_2
                     R : Type u_3
                     S : Type u_4
                     A : Type u_5
                     inst✝⁴ : CommRing K
                     inst✝³ : Field L
                     inst✝² : Algebra K L
                     inst✝¹ : NoZeroSMulDivisors K L
                     inst✝ : Algebra.IsAlgebraic K L
                     x✝ : AlgEquiv K L L
                     ⊢ Eq ((fun ϕ => AlgEquiv.ofBijective ϕ ⋯) ((fun ϕ => ↑ϕ) x✝)) x✝
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      K : Type u_1
                      L : Type u_2
                      R : Type u_3
                      S : Type u_4
                      A : Type u_5
                      inst✝⁴ : CommRing K
                      inst✝³ : Field L
                      inst✝² : Algebra K L
                      inst✝¹ : NoZeroSMulDivisors K L
                      inst✝ : Algebra.IsAlgebraic K L
                      x✝ : AlgHom K L L
                      ⊢ Eq ((fun ϕ => ↑ϕ) ((fun ϕ => AlgEquiv.ofBijective ϕ ⋯) x✝)) x✝
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/
  map_mul' _ _ := rfl


theorem IsAlgebraic.exists_nonzero_coeff_and_aeval_eq_zero
    {s : S} (hRs : IsAlgebraic R s) (hs : s ∈ nonZeroDivisors S) :
    ∃ q : R[X], q.coeff 0 ≠ 0 ∧ aeval s q = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    ⊢ Exists fun q => And (Ne (q.coeff 0) 0) (Eq ((Polynomial.aeval s) q) 0)
  -/
  obtain ⟨p, hp0, hp⟩ := hRs
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hs : Membership.mem (nonZeroDivisors S) s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ Exists fun q => And (Ne (q.coeff 0) 0) (Eq ((Polynomial.aeval s) q) 0)
  -/
  obtain ⟨q, hpq, hq⟩ := exists_eq_pow_rootMultiplicity_mul_and_not_dvd p hp0 0
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hs : Membership.mem (nonZeroDivisors S) s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    q : Polynomial R
    hpq : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 0)) (Po …
    hq : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 0)) q)
    ⊢ Exists fun q => And (Ne (q.coeff 0) 0) (Eq ((Polynomial.aeval s) q) 0)
  -/
  simp only [C_0, sub_zero, X_pow_mul, X_dvd_iff] at hpq hq
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hs : Membership.mem (nonZeroDivisors S) s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    q : Polynomial R
    hpq : Eq p (HMul.hMul q (HPow.hPow Polynomial.X (Polynomial.rootMultiplicity 0 …
    hq : Not (Eq (q.coeff 0) 0)
    ⊢ Exists fun q => And (Ne (q.coeff 0) 0) (Eq ((Polynomial.aeval s) q) 0)
  -/
  rw [hpq, map_mul, aeval_X_pow] at hp
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hs : Membership.mem (nonZeroDivisors S) s
    p : Polynomial R
    hp0 : Ne p 0
    q : Polynomial R
    hp : Eq (HMul.hMul ((Polynomial.aeval s) q) (HPow.hPow s (Polynomial.rootMulti …
    hpq : Eq p (HMul.hMul q (HPow.hPow Polynomial.X (Polynomial.rootMultiplicity 0 …
    hq : Not (Eq (q.coeff 0) 0)
    ⊢ Exists fun q => And (Ne (q.coeff 0) 0) (Eq ((Polynomial.aeval s) q) 0)
  -/
  exact ⟨q, hq, (nonZeroDivisors S).pow_mem hs (rootMultiplicity 0 p) (aeval s q) hp⟩
  /-
    🎉 no goals
  -/


theorem IsAlgebraic.exists_nonzero_eq_adjoin_mul
    {s : S} (hRs : IsAlgebraic R s) (hs : s ∈ nonZeroDivisors S) :
    ∃ᵉ (t ∈ Algebra.adjoin R {s}) (r ≠ (0 : R)), s * t = algebraMap R S r := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    ⊢ Exists fun t => And (Membership.mem (Algebra.adjoin R (Singleton.singleton s …
  -/
  have ⟨q, hq0, hq⟩ := hRs.exists_nonzero_coeff_and_aeval_eq_zero hs
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    ⊢ Exists fun t => And (Membership.mem (Algebra.adjoin R (Singleton.singleton s …
  -/
  have ⟨p, hp⟩ := X_dvd_sub_C (p := q)
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    p : Polynomial R
    hp : Eq (HSub.hSub q (Polynomial.C (q.coeff 0))) (HMul.hMul Polynomial.X p)
    ⊢ Exists fun t => And (Membership.mem (Algebra.adjoin R (Singleton.singleton s …
  -/
  refine ⟨aeval s p, aeval_mem_adjoin_singleton _ _, _, neg_ne_zero.mpr hq0, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    p : Polynomial R
    hp : Eq (HSub.hSub q (Polynomial.C (q.coeff 0))) (HMul.hMul Polynomial.X p)
    ⊢ Eq (HMul.hMul s ((Polynomial.aeval s) p)) ((algebraMap R S) (Neg.neg (q.coef …
  -/
  apply_fun aeval s at hp
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) (HSub.hSub q (Polynomial.C (q.coeff 0)))) ((Poly …
    ⊢ Eq (HMul.hMul s ((Polynomial.aeval s) p)) ((algebraMap R S) (Neg.neg (q.coef …
  -/
  rwa [map_sub, hq, zero_sub, map_mul, aeval_X, aeval_C, ← map_neg, eq_comm] at hp
  /-
    🎉 no goals
  -/


theorem IsAlgebraic.exists_nonzero_dvd
    {s : S} (hRs : IsAlgebraic R s) (hs : s ∈ nonZeroDivisors S) :
    ∃ r : R, r ≠ 0 ∧ s ∣ algebraMap R S r := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    ⊢ Exists fun r => And (Ne r 0) (Dvd.dvd s ((algebraMap R S) r))
  -/
  obtain ⟨q, hq0, hq⟩ := hRs.exists_nonzero_coeff_and_aeval_eq_zero hs
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    ⊢ Exists fun r => And (Ne r 0) (Dvd.dvd s ((algebraMap R S) r))
  -/
  have key := map_dvd (aeval s) (X_dvd_sub_C (p := q))
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    key : Dvd.dvd ((Polynomial.aeval s) Polynomial.X) ((Polynomial.aeval s) (HSub. …
    ⊢ Exists fun r => And (Ne r 0) (Dvd.dvd s ((algebraMap R S) r))
  -/
  rw [map_sub, hq, zero_sub, dvd_neg, aeval_X, aeval_C] at key
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : S
    hRs : IsAlgebraic R s
    hs : Membership.mem (nonZeroDivisors S) s
    q : Polynomial R
    hq0 : Ne (q.coeff 0) 0
    hq : Eq ((Polynomial.aeval s) q) 0
    key : Dvd.dvd s ((algebraMap R S) (q.coeff 0))
    ⊢ Exists fun r => And (Ne r 0) (Dvd.dvd s ((algebraMap R S) r))
  -/
  exact ⟨q.coeff 0, hq0, key⟩
  /-
    🎉 no goals
  -/


/-- A fraction `(a : S) / (b : S)` can be reduced to `(c : S) / (d : R)`,
if `b` is algebraic over `R`. -/
theorem IsAlgebraic.exists_smul_eq_mul
    (a : S) {b : S} (hRb : IsAlgebraic R b) (hb : b ∈ nonZeroDivisors S) :
    ∃ᵉ (c : S) (d ≠ (0 : R)), d • a = b * c :=
  have ⟨r, hr, s, h⟩ := hRb.exists_nonzero_dvd hb
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝² : CommRing R
                      inst✝¹ : Ring S
                      inst✝ : Algebra R S
                      a b : S
                      hRb : IsAlgebraic R b
                      hb : Membership.mem (nonZeroDivisors S) b
                      r : R
                      hr : Ne r 0
                      s : S
                      h : Eq ((algebraMap R S) r) (HMul.hMul b s)
                      ⊢ Eq (HSMul.hSMul r a) (HMul.hMul b (HMul.hMul s a))
                    -/
  ⟨s * a, r, hr, by rw [Algebra.smul_def, h, mul_assoc]⟩
                    /-
                      🎉 no goals
                    -/


/-- A fraction `(a : S) / (b : S)` can be reduced to `(c : S) / (d : R)`,
if `b` is algebraic over `R`. -/
theorem Algebra.IsAlgebraic.exists_smul_eq_mul [NoZeroDivisors S] [Algebra.IsAlgebraic R S]
    (a : S) {b : S} (hb : b ≠ 0) :
    ∃ᵉ (c : S) (d ≠ (0 : R)), d • a = b * c :=
  (isAlgebraic b).exists_smul_eq_mul a (mem_nonZeroDivisors_of_ne_zero hb)


theorem inv_eq_of_aeval_divX_ne_zero {x : L} {p : K[X]} (aeval_ne : aeval x (divX p) ≠ 0) :
    x⁻¹ = aeval x (divX p) / (aeval x p - algebraMap _ _ (p.coeff 0)) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_ne : Ne ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq (Inv.inv x) (HDiv.hDiv ((Polynomial.aeval x) p.divX) (HSub.hSub ((Polynom …
  -/
  rw [inv_eq_iff_eq_inv, inv_div, eq_comm, div_eq_iff, sub_eq_iff_eq_add, mul_comm]
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_ne : Ne ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq ((Polynomial.aeval x) p) (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) p.di …
  -/
  conv_lhs => rw [← divX_mul_X_add p]
    /-
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      x : L
      p : Polynomial K
      aeval_ne : Ne ((Polynomial.aeval x) p.divX) 0
      ⊢ Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul p.divX Polynomial.X) (Polynom …
    -/
  · rw [map_add, map_mul, aeval_X, aeval_C]
    /-
      🎉 no goals
    -/
    /-
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      x : L
      p : Polynomial K
      aeval_ne : Ne ((Polynomial.aeval x) p.divX) 0
      ⊢ Ne ((Polynomial.aeval x) p.divX) 0
    -/
  · exact aeval_ne
    /-
      🎉 no goals
    -/


theorem inv_eq_of_root_of_coeff_zero_ne_zero {x : L} {p : K[X]} (aeval_eq : aeval x p = 0)
    (coeff_zero_ne : p.coeff 0 ≠ 0) : x⁻¹ = -(aeval x (divX p) / algebraMap _ _ (p.coeff 0)) := by
  convert inv_eq_of_aeval_divX_ne_zero (p := p) (L := L)
    (mt (fun h => (algebraMap K L).injective ?_) coeff_zero_ne) using 1
    /-
      case h.e'_3
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      x : L
      p : Polynomial K
      aeval_eq : Eq ((Polynomial.aeval x) p) 0
      coeff_zero_ne : Ne (p.coeff 0) 0
      ⊢ Eq (Neg.neg (HDiv.hDiv ((Polynomial.aeval x) p.divX) ((algebraMap K L) (p.co …
    -/
  · rw [aeval_eq, zero_sub, div_neg]
    /-
      🎉 no goals
    -/
  /-
    case convert_2
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    h : Eq ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq ((algebraMap K L) (p.coeff 0)) ((algebraMap K L) 0)
  -/
  rw [RingHom.map_zero]
  /-
    case convert_2
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    h : Eq ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq ((algebraMap K L) (p.coeff 0)) 0
  -/
  convert aeval_eq
  /-
    case h.e'_2
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    h : Eq ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq ((algebraMap K L) (p.coeff 0)) ((Polynomial.aeval x) p)
  -/
  conv_rhs => rw [← divX_mul_X_add p]
  /-
    case h.e'_2
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    h : Eq ((Polynomial.aeval x) p.divX) 0
    ⊢ Eq ((algebraMap K L) (p.coeff 0)) ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMu …
  -/
  rw [map_add, map_mul, h, zero_mul, zero_add, aeval_C]
  /-
    🎉 no goals
  -/


theorem Subalgebra.inv_mem_of_root_of_coeff_zero_ne_zero {x : A} {p : K[X]}
    (aeval_eq : aeval x p = 0) (coeff_zero_ne : p.coeff 0 ≠ 0) : (x⁻¹ : L) ∈ A := by
  suffices (x⁻¹ : L) = (-p.coeff 0)⁻¹ • aeval x (divX p) by
    rw [this]
    exact A.smul_mem (aeval x _).2 _
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    ⊢ Eq (Inv.inv ↑x) (HSMul.hSMul (Inv.inv (Neg.neg (p.coeff 0))) ↑((Polynomial.a …
  -/
  have : aeval (x : L) p = 0 := by rw [Subalgebra.aeval_coe, aeval_eq, Subalgebra.coe_zero]
  -- Porting note: this was a long sequence of `rw`.
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    this : Eq ((Polynomial.aeval ↑x) p) 0
    ⊢ Eq (Inv.inv ↑x) (HSMul.hSMul (Inv.inv (Neg.neg (p.coeff 0))) ↑((Polynomial.a …
  -/
  rw [inv_eq_of_root_of_coeff_zero_ne_zero this coeff_zero_ne, div_eq_inv_mul, Algebra.smul_def]
  simp only [aeval_coe, Submonoid.coe_mul, Subsemiring.coe_toSubmonoid, coe_toSubsemiring,
    coe_algebraMap]
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    coeff_zero_ne : Ne (p.coeff 0) 0
    this : Eq ((Polynomial.aeval ↑x) p) 0
    ⊢ Eq (Neg.neg (HMul.hMul (Inv.inv ((algebraMap K L) (p.coeff 0))) ↑((Polynomia …
  -/
  rw [map_inv₀, map_neg, inv_neg, neg_mul]
  /-
    🎉 no goals
  -/


theorem Subalgebra.inv_mem_of_algebraic {x : A} (hx : _root_.IsAlgebraic K (x : L)) :
    (x⁻¹ : L) ∈ A := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    hx : _root_.IsAlgebraic K ↑x
    ⊢ Membership.mem A (Inv.inv ↑x)
  -/
  obtain ⟨p, ne_zero, aeval_eq⟩ := hx
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    ne_zero : Ne p 0
    aeval_eq : Eq ((Polynomial.aeval ↑x) p) 0
    ⊢ Membership.mem A (Inv.inv ↑x)
  -/
  rw [Subalgebra.aeval_coe, Subalgebra.coe_eq_zero] at aeval_eq
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    ne_zero : Ne p 0
    aeval_eq : Eq ((Polynomial.aeval x) p) 0
    ⊢ Membership.mem A (Inv.inv ↑x)
  -/
  revert ne_zero aeval_eq
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    A : Subalgebra K L
    x : Subtype fun x => Membership.mem A x
    p : Polynomial K
    ⊢ Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
  -/
  refine p.recOnHorner ?_ ?_ ?_
    /-
      case intro.intro.refine_1
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p : Polynomial K
      ⊢ Ne 0 0 → Eq ((Polynomial.aeval x) 0) 0 → Membership.mem A (Inv.inv ↑x)
    -/
  · intro h
    /-
      case intro.intro.refine_1
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p : Polynomial K
      h : Ne 0 0
      ⊢ Eq ((Polynomial.aeval x) 0) 0 → Membership.mem A (Inv.inv ↑x)
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p : Polynomial K
      ⊢ ∀ (p : Polynomial K) (a : K), Eq (p.coeff 0) 0 → Ne a 0 → (Ne p 0 → Eq ((Pol …
    -/
  · intro p a hp ha _ih _ne_zero aeval_eq
    /-
      case intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p✝ p : Polynomial K
      a : K
      hp : Eq (p.coeff 0) 0
      ha : Ne a 0
      _ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
      _ne_zero : Ne (HAdd.hAdd p (Polynomial.C a)) 0
      aeval_eq : Eq ((Polynomial.aeval x) (HAdd.hAdd p (Polynomial.C a))) 0
      ⊢ Membership.mem A (Inv.inv ↑x)
    -/
    refine A.inv_mem_of_root_of_coeff_zero_ne_zero aeval_eq ?_
    /-
      case intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p✝ p : Polynomial K
      a : K
      hp : Eq (p.coeff 0) 0
      ha : Ne a 0
      _ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
      _ne_zero : Ne (HAdd.hAdd p (Polynomial.C a)) 0
      aeval_eq : Eq ((Polynomial.aeval x) (HAdd.hAdd p (Polynomial.C a))) 0
      ⊢ Ne ((HAdd.hAdd p (Polynomial.C a)).coeff 0) 0
    -/
    rwa [coeff_add, hp, zero_add, coeff_C, if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p : Polynomial K
      ⊢ ∀ (p : Polynomial K), Ne p 0 → (Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Mem …
    -/
  · intro p hp ih _ne_zero aeval_eq
    /-
      case intro.intro.refine_3
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p✝ p : Polynomial K
      hp : Ne p 0
      ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
      _ne_zero : Ne (HMul.hMul p Polynomial.X) 0
      aeval_eq : Eq ((Polynomial.aeval x) (HMul.hMul p Polynomial.X)) 0
      ⊢ Membership.mem A (Inv.inv ↑x)
    -/
    rw [map_mul, aeval_X, mul_eq_zero] at aeval_eq
    /-
      case intro.intro.refine_3
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      A : Subalgebra K L
      x : Subtype fun x => Membership.mem A x
      p✝ p : Polynomial K
      hp : Ne p 0
      ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
      _ne_zero : Ne (HMul.hMul p Polynomial.X) 0
      aeval_eq : Or (Eq ((Polynomial.aeval x) p) 0) (Eq x 0)
      ⊢ Membership.mem A (Inv.inv ↑x)
    -/
    cases' aeval_eq with aeval_eq x_eq
      /-
        case intro.intro.refine_3.inl
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        A : Subalgebra K L
        x : Subtype fun x => Membership.mem A x
        p✝ p : Polynomial K
        hp : Ne p 0
        ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
        _ne_zero : Ne (HMul.hMul p Polynomial.X) 0
        aeval_eq : Eq ((Polynomial.aeval x) p) 0
        ⊢ Membership.mem A (Inv.inv ↑x)
      -/
    · exact ih hp aeval_eq
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_3.inr
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        A : Subalgebra K L
        x : Subtype fun x => Membership.mem A x
        p✝ p : Polynomial K
        hp : Ne p 0
        ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
        _ne_zero : Ne (HMul.hMul p Polynomial.X) 0
        x_eq : Eq x 0
        ⊢ Membership.mem A (Inv.inv ↑x)
      -/
    · rw [x_eq, Subalgebra.coe_zero, inv_zero]
      /-
        case intro.intro.refine_3.inr
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        A : Subalgebra K L
        x : Subtype fun x => Membership.mem A x
        p✝ p : Polynomial K
        hp : Ne p 0
        ih : Ne p 0 → Eq ((Polynomial.aeval x) p) 0 → Membership.mem A (Inv.inv ↑x)
        _ne_zero : Ne (HMul.hMul p Polynomial.X) 0
        x_eq : Eq x 0
        ⊢ Membership.mem A 0
      -/
      exact A.zero_mem
      /-
        🎉 no goals
      -/


/-- In an algebraic extension L/K, an intermediate subalgebra is a field. -/
@[stacks 0BID]
theorem Subalgebra.isField_of_algebraic [Algebra.IsAlgebraic K L] : IsField A :=
                         /-
                           K : Type u_1
                           L : Type u_2
                           inst✝³ : Field K
                           inst✝² : Field L
                           inst✝¹ : Algebra K L
                           A : Subalgebra K L
                           inst✝ : Algebra.IsAlgebraic K L
                           ⊢ Nontrivial (Subtype fun x => Membership.mem A x)
                         -/
  { show Nontrivial A by infer_instance, Subalgebra.toCommRing A with
                         /-
                           🎉 no goals
                         -/
    mul_inv_cancel := fun {a} ha =>
      ⟨⟨a⁻¹, A.inv_mem_of_algebraic (Algebra.IsAlgebraic.isAlgebraic (a : L))⟩,
        Subtype.ext (mul_inv_cancel₀ (mt (Subalgebra.coe_eq_zero _).mp ha))⟩ }


theorem Transcendental.infinite {x : A} (hx : Transcendental R x) : Infinite A :=
  .of_injective _ (transcendental_iff_injective.mp hx)


variable (R A) in
theorem Algebra.Transcendental.infinite [Algebra.Transcendental R A] : Infinite A :=
  have ⟨x, hx⟩ := ‹Algebra.Transcendental R A›
  hx.infinite


