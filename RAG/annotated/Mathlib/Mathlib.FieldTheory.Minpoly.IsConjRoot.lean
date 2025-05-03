variable (R) in
/--
We say that `y` is a conjugate root of `x` over `K` if the minimal polynomial of `x` is the
same as the minimal polynomial of `y`.
-/
def IsConjRoot (x y : A) : Prop := minpoly R x = minpoly R y


/--
The definition of conjugate roots.
-/
theorem isConjRoot_def {x y : A} : IsConjRoot R x y ↔ minpoly R x = minpoly R y := Iff.rfl


/--
Every element is a conjugate root of itself.
-/
@[refl] theorem refl {x : A} : IsConjRoot R x x := rfl


/--
If `y` is a conjugate root of `x`, then `x` is also a conjugate root of `y`.
-/
@[symm] theorem symm {x y : A} (h : IsConjRoot R x y) : IsConjRoot R y x := Eq.symm h


/--
If `y` is a conjugate root of `x` and `z` is a conjugate root of `y`, then `z` is a conjugate
root of `x`.
-/
@[trans] theorem trans {x y z: A} (h₁ : IsConjRoot R x y) (h₂ : IsConjRoot R y z) :
    IsConjRoot R x z := Eq.trans h₁ h₂


variable (R A) in
/--
The setoid structure on `A` defined by the equivalence relation of `IsConjRoot R · ·`.
-/
def setoid : Setoid A where
  r := IsConjRoot R
  iseqv := ⟨fun _ => refl, symm, trans⟩


/--
Let `p` be the minimal polynomial of `x`. If `y` is a conjugate root of `x`, then `p y = 0`.
-/
theorem aeval_eq_zero {x y : A} (h : IsConjRoot R x y) : aeval y (minpoly R x) = 0 :=
  h ▸ minpoly.aeval R y


/--
Let `r` be an element of the base ring. If `y` is a conjugate root of `x`, then `y + r` is a
conjugate root of `x + r`.
-/
theorem add_algebraMap {x y : S} (r : K) (h : IsConjRoot K x y) :
    IsConjRoot K (x + algebraMap K S r) (y + algebraMap K S r) := by
  /-
    K : Type u_2
    S : Type u_4
    inst✝² : CommRing S
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x y : S
    r : K
    h : IsConjRoot K x y
    ⊢ IsConjRoot K (HAdd.hAdd x ((algebraMap K S) r)) (HAdd.hAdd y ((algebraMap K  …
  -/
  rw [isConjRoot_def, minpoly.add_algebraMap x r, minpoly.add_algebraMap y r, h]
  /-
    🎉 no goals
  -/


/--
Let `r` be an element of the base ring. If `y` is a conjugate root of `x`, then `y - r` is a
conjugate root of `x - r`.
-/
theorem sub_algebraMap {x y : S} (r : K) (h : IsConjRoot K x y) :
    IsConjRoot K (x - algebraMap K S r) (y - algebraMap K S r) := by
  /-
    K : Type u_2
    S : Type u_4
    inst✝² : CommRing S
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x y : S
    r : K
    h : IsConjRoot K x y
    ⊢ IsConjRoot K (HSub.hSub x ((algebraMap K S) r)) (HSub.hSub y ((algebraMap K  …
  -/
  simpa only [sub_eq_add_neg, map_neg] using add_algebraMap (-r) h
  /-
    🎉 no goals
  -/


/--
If `y` is a conjugate root of `x`, then `-y` is a conjugate root of `-x`.
-/
theorem neg {x y : S} (h : IsConjRoot K x y) :
    IsConjRoot K (-x) (-y) := by
  /-
    K : Type u_2
    S : Type u_4
    inst✝² : CommRing S
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x y : S
    h : IsConjRoot K x y
    ⊢ IsConjRoot K (Neg.neg x) (Neg.neg y)
  -/
  rw [isConjRoot_def, minpoly.neg x, minpoly.neg y, h]
  /-
    🎉 no goals
  -/


/--
A variant of `isConjRoot_algHom_iff`, only assuming `Function.Injective f`,
instead of `DivisionRing A`.
If `y` is a conjugate root of `x` and `f` is an injective `R`-algebra homomorphism, then `f y` is
a conjugate root of `f x`.
-/
theorem isConjRoot_algHom_iff_of_injective {x y : A} {f : A →ₐ[R] B}
    (hf : Function.Injective f) : IsConjRoot R (f x) (f y) ↔ IsConjRoot R x y := by
  /-
    R : Type u_1
    A : Type u_5
    B : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    x y : A
    f : AlgHom R A B
    hf : Function.Injective ⇑f
    ⊢ Iff (IsConjRoot R (f x) (f y)) (IsConjRoot R x y)
  -/
  rw [isConjRoot_def, isConjRoot_def, algHom_eq f hf, algHom_eq f hf]
  /-
    🎉 no goals
  -/


/--
If `y` is a conjugate root of `x` in some division ring and `f` is a `R`-algebra homomorphism, then
`f y` is a conjugate root of `f x`.
-/
theorem isConjRoot_algHom_iff {A} [DivisionRing A] [Algebra R A]
    [Nontrivial B] {x y : A} (f : A →ₐ[R] B) : IsConjRoot R (f x) (f y) ↔ IsConjRoot R x y :=
  isConjRoot_algHom_iff_of_injective f.injective


/--
Let `p` be the minimal polynomial of an integral element `x`. If `p y` = 0, then `y` is a
conjugate root of `x`.
-/
theorem isConjRoot_of_aeval_eq_zero [IsDomain A] {x y : A} (hx : IsIntegral K x)
    (h : aeval y (minpoly K x) = 0) : IsConjRoot K x y :=
  minpoly.eq_of_irreducible_of_monic (minpoly.irreducible hx) h (minpoly.monic hx)


/--
Let `p` be the minimal polynomial of an integral element `x`. Then `y` is a conjugate root of `x`
if and only if `p y = 0`.
-/
theorem isConjRoot_iff_aeval_eq_zero [IsDomain A] {x y : A}
    (h : IsIntegral K x) : IsConjRoot K x y ↔ aeval y (minpoly K x) = 0 :=
  ⟨IsConjRoot.aeval_eq_zero, isConjRoot_of_aeval_eq_zero h⟩


/--
Let `s` be an `R`-algebra isomorphism. Then `s x` is a conjugate root of `x`.
-/
@[simp]
theorem isConjRoot_of_algEquiv (x : A) (s : A ≃ₐ[R] A) : IsConjRoot R x (s x) :=
  Eq.symm (minpoly.algEquiv_eq s x)


/--
A variant of `isConjRoot_of_algEquiv`.
Let `s` be an `R`-algebra isomorphism. Then `x` is a conjugate root of `s x`.
-/
@[simp]
theorem isConjRoot_of_algEquiv' (x : A) (s : A ≃ₐ[R] A) : IsConjRoot R (s x) x :=
  (minpoly.algEquiv_eq s x)


/--
Let `s₁` and `s₂` be two `R`-algebra isomorphisms. Then `s₂ x` is a conjugate root of `s₁ x`.
-/
@[simp]
theorem isConjRoot_of_algEquiv₂ (x : A) (s₁ s₂ : A ≃ₐ[R] A) : IsConjRoot R (s₁ x) (s₂ x) :=
  isConjRoot_def.mpr <| (minpoly.algEquiv_eq s₂ x) ▸ (minpoly.algEquiv_eq s₁ x)


/--
Let `L / K` be a normal field extension. For any two elements `x` and `y` in `L`, if `y` is a
conjugate root of `x`, then there exists a `K`-automorphism `σ : L ≃ₐ[K] L` such
that `σ y = x`.
-/
theorem IsConjRoot.exists_algEquiv [Normal K L] {x y: L} (h : IsConjRoot K x y) :
    ∃ σ : L ≃ₐ[K] L, σ y = x := by
  obtain ⟨σ, hσ⟩ :=
    exists_algHom_of_splits_of_aeval (normal_iff.mp inferInstance) (h ▸ minpoly.aeval K x)
  /-
    case intro
    K : Type u_2
    L : Type u_3
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    h : IsConjRoot K x y
    σ : AlgHom K L L
    hσ : Eq (σ y) x
    ⊢ Exists fun σ => Eq (σ y) x
  -/
  exact ⟨AlgEquiv.ofBijective σ (σ.normal_bijective _ _ _), hσ⟩
  /-
    🎉 no goals
  -/


/--
Let `L / K` be a normal field extension. For any two elements `x` and `y` in `L`, `y` is a
conjugate root of `x` if and only if there exists a `K`-automorphism `σ : L ≃ₐ[K] L` such
that `σ y = x`.
-/
theorem isConjRoot_iff_exists_algEquiv [Normal K L] {x y : L} :
    IsConjRoot K x y ↔ ∃ σ : L ≃ₐ[K] L, σ y = x :=
  ⟨exists_algEquiv, fun ⟨_, h⟩ => h ▸ (isConjRoot_of_algEquiv _ _).symm⟩


/--
Let `L / K` be a normal field extension. For any two elements `x` and `y` in `L`, `y` is a
conjugate root of `x` if and only if `x` and `y` falls in the same orbit of the action of Galois
group.
-/
theorem isConjRoot_iff_orbitRel [Normal K L] {x y : L} :
    IsConjRoot K x y ↔ MulAction.orbitRel (L ≃ₐ[K] L) L x y:=
  (isConjRoot_iff_exists_algEquiv)


/--
Let `S / L / K` be a tower of extensions. For any two elements `y` and `x` in `S`, if `y` is a
conjugate root of `x` over `L`, then `y` is also a conjugate root of `x` over
`K`.
-/
theorem IsConjRoot.of_isScalarTower [IsScalarTower K L S] {x y : S} (hx : IsIntegral K x)
    (h : IsConjRoot L x y) : IsConjRoot K x y :=
  isConjRoot_of_aeval_eq_zero hx <| minpoly.aeval_of_isScalarTower K x y (aeval_eq_zero h)


/--
`y` is a conjugate root of `x` over `K` if and only if `y` is a root of the minimal polynomial of
`x`. This is variant of `isConjRoot_iff_aeval_eq_zero`.
-/
theorem isConjRoot_iff_mem_minpoly_aroots {x y : S} (h : IsIntegral K x) :
    IsConjRoot K x y ↔ y ∈ (minpoly K x).aroots S := by
  /-
    K : Type u_2
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : IsDomain S
    x y : S
    h : IsIntegral K x
    ⊢ Iff (IsConjRoot K x y) (Membership.mem ((minpoly K x).aroots S) y)
  -/
  rw [Polynomial.mem_aroots, isConjRoot_iff_aeval_eq_zero h]
  /-
    K : Type u_2
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : IsDomain S
    x y : S
    h : IsIntegral K x
    ⊢ Iff (Eq ((Polynomial.aeval y) (minpoly K x)) 0) (And (Ne (minpoly K x) 0) (E …
  -/
  simp only [iff_and_self]
  /-
    K : Type u_2
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : IsDomain S
    x y : S
    h : IsIntegral K x
    ⊢ Eq ((Polynomial.aeval y) (minpoly K x)) 0 → Ne (minpoly K x) 0
  -/
  exact fun _ => minpoly.ne_zero h
  /-
    🎉 no goals
  -/


/--
`y` is a conjugate root of `x` over `K` if and only if `y` is a root of the minimal polynomial of
`x`. This is variant of `isConjRoot_iff_aeval_eq_zero`.
-/
theorem isConjRoot_iff_mem_minpoly_rootSet {x y : S}
    (h : IsIntegral K x) : IsConjRoot K x y ↔ y ∈ (minpoly K x).rootSet S :=
                                                  /-
                                                    K : Type u_2
                                                    S : Type u_4
                                                    inst✝³ : CommRing S
                                                    inst✝² : Field K
                                                    inst✝¹ : Algebra K S
                                                    inst✝ : IsDomain S
                                                    x y : S
                                                    h : IsIntegral K x
                                                    ⊢ Iff (Membership.mem ((minpoly K x).aroots S) y) (Membership.mem ((minpoly K  …
                                                  -/
  (isConjRoot_iff_mem_minpoly_aroots h).trans (by simp [rootSet])
                                                  /-
                                                    🎉 no goals
                                                  -/


/--
If `y` is a conjugate root of an integral element `x` over `R`, then `y` is also integral
over `R`.
-/
theorem isIntegral {x y : A} (hx : IsIntegral R x) (h : IsConjRoot R x y) :
    IsIntegral R y :=
  ⟨minpoly R x, minpoly.monic hx, h ▸ minpoly.aeval R y⟩


/--
A variant of `IsConjRoot.eq_of_isConjRoot_algebraMap`, only assuming `Nontrivial R`,
`NoZeroSMulDivisors R A` and `Function.Injective (algebraMap R A)` instead of `Field R`. If `x` is a
conjugate root of some element `algebraMap R S r` in the image of the base ring, then
`x = algebraMap R S r`.
-/
theorem eq_algebraMap_of_injective [Nontrivial R] [NoZeroSMulDivisors R S] {r : R} {x : S}
    (h : IsConjRoot R (algebraMap R S r) x) (hf : Function.Injective (algebraMap R S)) :
    x = algebraMap R S r := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroSMulDivisors R S
    r : R
    x : S
    h : IsConjRoot R ((algebraMap R S) r) x
    hf : Function.Injective ⇑(algebraMap R S)
    ⊢ Eq x ((algebraMap R S) r)
  -/
  rw [IsConjRoot, minpoly.eq_X_sub_C_of_algebraMap_inj _ hf] at h
  have : x ∈ (X - C r).aroots S := by
    rw [mem_aroots]
    simp [X_sub_C_ne_zero, h ▸ minpoly.aeval R x]
  /-
    R : Type u_1
    S : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroSMulDivisors R S
    r : R
    x : S
    h : Eq (HSub.hSub Polynomial.X (Polynomial.C r)) (minpoly R x)
    hf : Function.Injective ⇑(algebraMap R S)
    this : Membership.mem ((HSub.hSub Polynomial.X (Polynomial.C r)).aroots S) x
    ⊢ Eq x ((algebraMap R S) r)
  -/
  simpa [aroots_X_sub_C] using this
  /-
    🎉 no goals
  -/


/--
If `x` is a conjugate root of some element `algebraMap R S r` in the image of the base ring, then
`x = algebraMap R S r`.
-/
theorem eq_algebraMap {r : K} {x : S} (h : IsConjRoot K (algebraMap K S r) x) :
    x = algebraMap K S r :=
  eq_algebraMap_of_injective h (algebraMap K S).injective


/--
A variant of `IsConjRoot.eq_zero`, only assuming `Nontrivial R`,
`NoZeroSMulDivisors R A` and `Function.Injective (algebraMap R A)` instead of `Field R`. If `x` is a
conjugate root of `0`, then `x = 0`.
-/
theorem eq_zero_of_injective [Nontrivial R] [NoZeroSMulDivisors R S] {x : S} (h : IsConjRoot R 0 x)
    (hf : Function.Injective (algebraMap R S)) : x = 0 :=
  (algebraMap R S).map_zero ▸ (eq_algebraMap_of_injective ((algebraMap R S).map_zero ▸ h) hf)


/--
If `x` is a conjugate root of `0`, then `x = 0`.
-/
theorem eq_zero {x : S} (h : IsConjRoot K 0 x) : x = 0 :=
  eq_zero_of_injective h (algebraMap K S).injective


/--
A variant of `IsConjRoot.eq_of_isConjRoot_algebraMap`, only assuming `Nontrivial R`,
`NoZeroSMulDivisors R A` and `Function.Injective (algebraMap R A)` instead of `Field R`. If `x` is a
conjugate root of some element `algebraMap R S r` in the image of the base ring, then
`x = algebraMap R S r`.
-/
theorem isConjRoot_iff_eq_algebraMap_of_injective [Nontrivial R] [NoZeroSMulDivisors R S] {r : R}
    {x : S} (hf : Function.Injective (algebraMap R S)) :
    IsConjRoot R (algebraMap R S r) x ↔ x = algebraMap R S r :=
    ⟨fun h => eq_algebraMap_of_injective h hf, fun h => h.symm ▸ rfl⟩


/--
An element `x` is a conjugate root of some element `algebraMap R S r` in the image of the base ring
if and only if `x = algebraMap R S r`.
-/
@[simp]
theorem isConjRoot_iff_eq_algebraMap {r : K} {x : S} :
    IsConjRoot K (algebraMap K S r) x ↔ x = algebraMap K S r :=
  isConjRoot_iff_eq_algebraMap_of_injective (algebraMap K S).injective


/--
A variant of `isConjRoot_iff_eq_algebraMap`.
an element `algebraMap R S r` in the image of the base ring is a conjugate root of an element `x`
if and only if `x = algebraMap R S r`.
-/
@[simp]
theorem isConjRoot_iff_eq_algebraMap' {r : K} {x : S} :
    IsConjRoot K x (algebraMap K S r) ↔ x = algebraMap K S r :=
  eq_comm.trans <| isConjRoot_iff_eq_algebraMap_of_injective (algebraMap K S).injective


/--
A variant of `IsConjRoot.iff_eq_zero`, only assuming `Nontrivial R`,
`NoZeroSMulDivisors R A` and `Function.Injective (algebraMap R A)` instead of `Field R`. `x` is a
conjugate root of `0` if and only if `x = 0`.
-/
theorem isConjRoot_zero_iff_eq_zero_of_injective [Nontrivial R] {x : S} [NoZeroSMulDivisors R S]
    (hf : Function.Injective (algebraMap R S)) : IsConjRoot R 0 x ↔ x = 0 :=
  ⟨fun h => eq_zero_of_injective h hf, fun h => h.symm ▸ rfl⟩


/--
`x` is a conjugate root of `0` if and only if `x = 0`.
-/
@[simp]
theorem isConjRoot_zero_iff_eq_zero {x : S} : IsConjRoot K 0 x ↔ x = 0 :=
  isConjRoot_zero_iff_eq_zero_of_injective (algebraMap K S).injective


/--
A variant of `IsConjRoot.iff_eq_zero`. `0` is a conjugate root of `x` if and only if `x = 0`.
-/
@[simp]
theorem isConjRoot_zero_iff_eq_zero' {x : S} : IsConjRoot K x 0 ↔ x = 0 :=
  eq_comm.trans <| isConjRoot_zero_iff_eq_zero_of_injective (algebraMap K S).injective


/--
A variant of `IsConjRoot.ne_zero`, only assuming `Nontrivial R`,
`NoZeroSMulDivisors R A` and `Function.Injective (algebraMap R A)` instead of `Field R`. If `y` is
a conjugate root of a nonzero element `x`, then `y` is not zero.
-/
theorem ne_zero_of_injective [Nontrivial R] [NoZeroSMulDivisors R S] {x y : S} (hx : x ≠ 0)
    (h : IsConjRoot R x y) (hf : Function.Injective (algebraMap R S)) : y ≠ 0 :=
  fun g => hx (eq_zero_of_injective (g ▸ h.symm) hf)


/--
If `y` is a conjugate root of a nonzero element `x`, then `y` is not zero.
-/
theorem ne_zero {x y : S} (hx : x ≠ 0) (h : IsConjRoot K x y) : y ≠ 0 :=
  ne_zero_of_injective hx h (algebraMap K S).injective


/--
Let `L / K` be a field extension. If `x` is a separable element over `K` and the minimal polynomial
of `x` splits in `L`, then `x` is not in `K` if and only if there exists a conjugate
root of `x` over `K` in `L` which is not equal to `x` itself.
-/
theorem not_mem_iff_exists_ne_and_isConjRoot {x : L} (h : IsSeparable K x)
    (sp : (minpoly K x).Splits (algebraMap K L)) :
    x ∉ (⊥ : Subalgebra K L) ↔ ∃ y : L, x ≠ y ∧ IsConjRoot K x y := by
  calc
    _ ↔ 2 ≤ (minpoly K x).natDegree := (minpoly.two_le_natDegree_iff h.isIntegral).symm
    _ ↔ 2 ≤ Fintype.card ((minpoly K x).rootSet L) :=
      (Polynomial.card_rootSet_eq_natDegree h sp) ▸ Iff.rfl
    _ ↔ Nontrivial ((minpoly K x).rootSet L) := Fintype.one_lt_card_iff_nontrivial
    _ ↔ ∃ y : ((minpoly K x).rootSet L), ↑y ≠ x :=
      (nontrivial_iff_exists_ne ⟨x, mem_rootSet.mpr ⟨minpoly.ne_zero h.isIntegral,
          minpoly.aeval K x⟩⟩).trans ⟨fun ⟨y, hy⟩ => ⟨y, Subtype.coe_ne_coe.mpr hy⟩,
          fun ⟨y, hy⟩ => ⟨y, Subtype.coe_ne_coe.mp hy⟩⟩
    _ ↔ _ :=
      ⟨fun ⟨⟨y, hy⟩, hne⟩ => ⟨y, ⟨hne.symm,
          (isConjRoot_iff_mem_minpoly_rootSet h.isIntegral).mpr hy⟩⟩,
          fun ⟨y, hne, hy⟩ => ⟨⟨y,
          (isConjRoot_iff_mem_minpoly_rootSet h.isIntegral).mp hy⟩, hne.symm⟩⟩

