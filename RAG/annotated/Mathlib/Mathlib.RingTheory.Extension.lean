/--
An extension of an `R`-algebra `S` is an `R` algebra `P` together with a surjection `P →ₐ[R] S`.
Also see `Algebra.Extension.ofSurjective`.
-/
structure Algebra.Extension where
  /-- The underlying algebra of an extension. -/
  Ring : Type w
  [commRing : CommRing Ring]
  [algebra₁ : Algebra R Ring]
  [algebra₂ : Algebra Ring S]
  [isScalarTower : IsScalarTower R Ring S]
  /-- A chosen (set-theoretic) section of an extension. -/
  σ : S → Ring
  algebraMap_σ : ∀ x, algebraMap Ring S (σ x) = x


@[nolint unusedArguments]
noncomputable instance {R₀} [CommRing R₀] [Algebra R₀ R] [Algebra R₀ S] [IsScalarTower R₀ R S] :
    Algebra R₀ P.Ring where
  __ := Module.compHom P.Ring (algebraMap R₀ R)
  __ := (algebraMap R P.Ring).comp (algebraMap R₀ R)
  smul_def' _ _ := smul_def _ _
  commutes' _ _ := commutes _ _


instance {R₀} [CommRing R₀] [Algebra R₀ R] [Algebra R₀ S] [IsScalarTower R₀ R S] :
    IsScalarTower R₀ R P.Ring := IsScalarTower.of_algebraMap_eq' rfl


instance {R₀} [CommRing R₀] [Algebra R₀ R] [Algebra R₀ S] [IsScalarTower R₀ R S]
    {R₁} [CommRing R₁] [Algebra R₁ R] [Algebra R₁ S] [IsScalarTower R₁ R S]
    [Algebra R₀ R₁] [IsScalarTower R₀ R₁ R] :
    IsScalarTower R₀ R₁ P.Ring := IsScalarTower.of_algebraMap_eq' <| by
  rw [IsScalarTower.algebraMap_eq R₀ R, IsScalarTower.algebraMap_eq R₁ R,
    RingHom.comp_assoc, ← IsScalarTower.algebraMap_eq R₀ R₁ R]


instance {R₀} [CommRing R₀] [Algebra R₀ R] [Algebra R₀ S] [IsScalarTower R₀ R S] :
    IsScalarTower R₀ P.Ring S := IsScalarTower.of_algebraMap_eq' <| by
  rw [IsScalarTower.algebraMap_eq R₀ R P.Ring, ← RingHom.comp_assoc,
    ← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]


@[simp]
lemma σ_smul (x y) : P.σ x • y = x * y := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    x y : S
    ⊢ Eq (HSMul.hSMul (P.σ x) y) (HMul.hMul x y)
  -/
  rw [Algebra.smul_def, algebraMap_σ]
  /-
    🎉 no goals
  -/


lemma σ_injective : P.σ.Injective := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    ⊢ Function.Injective P.σ
  -/
  intro x y e
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    x y : S
    e : Eq (P.σ x) (P.σ y)
    ⊢ Eq x y
  -/
  rw [← P.algebraMap_σ x, ← P.algebraMap_σ y, e]
  /-
    🎉 no goals
  -/


lemma algebraMap_surjective : Function.Surjective (algebraMap P.Ring S) := (⟨_, P.algebraMap_σ ·⟩)


/-- Construct `Extension` from a surjective algebra homomorphism. -/
@[simps (config := .lemmasOnly) Ring σ]
noncomputable
def ofSurjective {P : Type w} [CommRing P] [Algebra R P] (f : P →ₐ[R] S)
    (h : Function.Surjective f) : Extension.{w} R S where
  Ring := P
  algebra₂ := f.toAlgebra
  isScalarTower := letI := f.toAlgebra; IsScalarTower.of_algebraMap_eq' f.comp_algebraMap.symm
  σ x := (h x).choose
  algebraMap_σ x := (h x).choose_spec


variable (R S) in
/-- The trivial extension of `S`. -/
@[simps (config := .lemmasOnly) Ring σ]
noncomputable
def self : Extension R S where
  Ring := S
  σ := _root_.id
  algebraMap_σ _ := rfl


/--
An `R`-extension `P → S` gives an `R`-extension `Pₘ → Sₘ`.
Note that this is different from `baseChange` as the base does not change.
-/
noncomputable
def localization (P : Extension.{w} R S) : Extension R S' where
  Ring := Localization (M.comap (algebraMap P.Ring S))
  algebra₂ := (IsLocalization.lift (M := (M.comap (algebraMap P.Ring S)))
      (g := (algebraMap S S').comp (algebraMap P.Ring S))
          /-
            R : Type u
            S : Type v
            inst✝⁷ : CommRing R
            inst✝⁶ : CommRing S
            inst✝⁵ : Algebra R S
            P✝ : Algebra.Extension R S
            M : Submonoid S
            S' : Type u_1
            inst✝⁴ : CommRing S'
            inst✝³ : Algebra S S'
            inst✝² : IsLocalization M S'
            inst✝¹ : Algebra R S'
            inst✝ : IsScalarTower R S S'
            P : Algebra.Extension R S
            ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.comap (algebraMap P.Ring S …
          -/
      (by simpa using fun x hx ↦ IsLocalization.map_units S' ⟨_, hx⟩)).toAlgebra
          /-
            🎉 no goals
          -/
  isScalarTower := by
    letI : Algebra (Localization (M.comap (algebraMap P.Ring S))) S' :=
      (IsLocalization.lift (M := (M.comap (algebraMap P.Ring S)))
        (g := (algebraMap S S').comp (algebraMap P.Ring S))
        (by simpa using fun x hx ↦ IsLocalization.map_units S' ⟨_, hx⟩)).toAlgebra
    /-
      R : Type u
      S : Type v
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      P✝ : Algebra.Extension R S
      M : Submonoid S
      S' : Type u_1
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra S S'
      inst✝² : IsLocalization M S'
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R S S'
      P : Algebra.Extension R S
      this : Algebra (Localization (Submonoid.comap (algebraMap P.Ring S) M)) S' :=  …
      ⊢ IsScalarTower R (Localization (Submonoid.comap (algebraMap P.Ring S) M)) S'
    -/
    apply IsScalarTower.of_algebraMap_eq'
    rw [RingHom.algebraMap_toAlgebra, IsScalarTower.algebraMap_eq R P.Ring (Localization _),
      ← RingHom.comp_assoc, IsLocalization.lift_comp, RingHom.comp_assoc,
      ← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]
                                                                                              /-
                                                                                                R : Type u
                                                                                                S : Type v
                                                                                                inst✝⁷ : CommRing R
                                                                                                inst✝⁶ : CommRing S
                                                                                                inst✝⁵ : Algebra R S
                                                                                                P✝ : Algebra.Extension R S
                                                                                                M : Submonoid S
                                                                                                S' : Type u_1
                                                                                                inst✝⁴ : CommRing S'
                                                                                                inst✝³ : Algebra S S'
                                                                                                inst✝² : IsLocalization M S'
                                                                                                inst✝¹ : Algebra R S'
                                                                                                inst✝ : IsScalarTower R S S'
                                                                                                P : Algebra.Extension R S
                                                                                                s : S'
                                                                                                ⊢ Membership.mem (Submonoid.comap (algebraMap P.Ring S) M) (P.σ ↑(IsLocalizati …
                                                                                              -/
  σ s := Localization.mk (P.σ (IsLocalization.sec M s).1) ⟨P.σ (IsLocalization.sec M s).2, by simp⟩
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
  algebraMap_σ s := by
    simp [RingHom.algebraMap_toAlgebra, Localization.mk_eq_mk', IsLocalization.lift_mk',
      Units.mul_inv_eq_iff_eq_mul, IsUnit.coe_liftRight, IsLocalization.sec_spec]


/-- The base change of an `R`-extension of `S` to `T` gives a `T`-extension of `T ⊗[R] S`. -/
noncomputable
def baseChange {T} [CommRing T] [Algebra R T] (P : Extension R S) : Extension T (T ⊗[R] S) where
  Ring := T ⊗[R] P.Ring
  __ := ofSurjective (P := T ⊗[R] P.Ring) (Algebra.TensorProduct.map (AlgHom.id T T)
    (IsScalarTower.toAlgHom _ _ _)) (LinearMap.lTensor_surjective T
    (g := (IsScalarTower.toAlgHom R P.Ring S).toLinearMap) P.algebraMap_surjective)



/-- Given a commuting square
```
R --→ P -→ S
|          |
↓          ↓
R' -→ P' → S
```
A hom between `P` and `P'` is a ring homomorphism that makes the two squares commute.
-/
@[ext]
structure Hom where
  /-- The underlying ring homomorphism of a hom between extensions. -/
  toRingHom : P.Ring →+* P'.Ring
  toRingHom_algebraMap :
    ∀ x, toRingHom (algebraMap R P.Ring x) = algebraMap R' P'.Ring (algebraMap R R' x)
  algebraMap_toRingHom :
    ∀ x, (algebraMap P'.Ring S' (toRingHom x)) = algebraMap S S' (algebraMap P.Ring S x)


/-- A hom between extensions as an algebra homomorphism. -/
noncomputable
def Hom.toAlgHom [Algebra R S'] [IsScalarTower R R' S'] (f : Hom P P') :
    P.Ring →ₐ[R] P'.Ring where
  __ := f.toRingHom
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹⁶ : CommRing R
                    inst✝¹⁵ : CommRing S
                    inst✝¹⁴ : Algebra R S
                    P : Algebra.Extension R S
                    R' : Type ?u.129704
                    S' : Type ?u.129707
                    inst✝¹³ : CommRing R'
                    inst✝¹² : CommRing S'
                    inst✝¹¹ : Algebra R' S'
                    P' : Algebra.Extension R' S'
                    R'' : Type ?u.130041
                    S'' : Type ?u.130044
                    inst✝¹⁰ : CommRing R''
                    inst✝⁹ : CommRing S''
                    inst✝⁸ : Algebra R'' S''
                    P'' : Algebra.Extension R'' S''
                    inst✝⁷ : Algebra R R'
                    inst✝⁶ : Algebra R' R''
                    inst✝⁵ : Algebra R R''
                    inst✝⁴ : Algebra S S'
                    inst✝³ : Algebra S' S''
                    inst✝² : Algebra S S''
                    inst✝¹ : Algebra R S'
                    inst✝ : IsScalarTower R R' S'
                    f : P.Hom P'
                    ⊢ ∀ (r : R), Eq ((↑↑__spread✝⁻⁰).toFun ((algebraMap R P.Ring) r)) ((algebraMap …
                  -/
  commutes' := by simp [← IsScalarTower.algebraMap_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma Hom.toAlgHom_apply [Algebra R S'] [IsScalarTower R R' S'] (f : Hom P P') (x) :
    f.toAlgHom x = f.toRingHom x := rfl


/-- The identity hom. -/
@[simps]
                                                                  /-
                                                                    R : Type u
                                                                    S : Type v
                                                                    inst✝¹⁴ : CommRing R
                                                                    inst✝¹³ : CommRing S
                                                                    inst✝¹² : Algebra R S
                                                                    P : Algebra.Extension R S
                                                                    R' : Type ?u.142485
                                                                    S' : Type ?u.142488
                                                                    inst✝¹¹ : CommRing R'
                                                                    inst✝¹⁰ : CommRing S'
                                                                    inst✝⁹ : Algebra R' S'
                                                                    P' : Algebra.Extension R' S'
                                                                    R'' : Type ?u.142822
                                                                    S'' : Type ?u.142825
                                                                    inst✝⁸ : CommRing R''
                                                                    inst✝⁷ : CommRing S''
                                                                    inst✝⁶ : Algebra R'' S''
                                                                    P'' : Algebra.Extension R'' S''
                                                                    inst✝⁵ : Algebra R R'
                                                                    inst✝⁴ : Algebra R' R''
                                                                    inst✝³ : Algebra R R''
                                                                    inst✝² : Algebra S S'
                                                                    inst✝¹ : Algebra S' S''
                                                                    inst✝ : Algebra S S''
                                                                    ⊢ ∀ (x : R), Eq ((RingHom.id P.Ring) ((algebraMap R P.Ring) x)) ((algebraMap R …
                                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
protected noncomputable def Hom.id : Hom P P := ⟨RingHom.id _, by simp, by simp⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
                                                                   /-
                                                                     R : Type u
                                                                     S : Type v
                                                                     inst✝² : CommRing R
                                                                     inst✝¹ : CommRing S
                                                                     inst✝ : Algebra R S
                                                                     P : Algebra.Extension R S
                                                                     ⊢ Eq (Algebra.Extension.Hom.id P).toAlgHom (AlgHom.id R P.Ring)
                                                                   -/
lemma Hom.toAlgHom_id : Hom.toAlgHom (.id P) = AlgHom.id _ _ := by ext1; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


variable [IsScalarTower R R' R''] [IsScalarTower S S' S''] in
/-- The composition of two homs. -/
@[simps]
noncomputable def Hom.comp (f : Hom P' P'') (g : Hom P P') : Hom P P'' where
  toRingHom := f.toRingHom.comp g.toRingHom
                             /-
                               R : Type u
                               S : Type v
                               inst✝¹⁶ : CommRing R
                               inst✝¹⁵ : CommRing S
                               inst✝¹⁴ : Algebra R S
                               P : Algebra.Extension R S
                               R' : Type ?u.156843
                               S' : Type ?u.156846
                               inst✝¹³ : CommRing R'
                               inst✝¹² : CommRing S'
                               inst✝¹¹ : Algebra R' S'
                               P' : Algebra.Extension R' S'
                               R'' : Type ?u.157180
                               S'' : Type ?u.157183
                               inst✝¹⁰ : CommRing R''
                               inst✝⁹ : CommRing S''
                               inst✝⁸ : Algebra R'' S''
                               P'' : Algebra.Extension R'' S''
                               inst✝⁷ : Algebra R R'
                               inst✝⁶ : Algebra R' R''
                               inst✝⁵ : Algebra R R''
                               inst✝⁴ : Algebra S S'
                               inst✝³ : Algebra S' S''
                               inst✝² : Algebra S S''
                               inst✝¹ : IsScalarTower R R' R''
                               inst✝ : IsScalarTower S S' S''
                               f : P'.Hom P''
                               g : P.Hom P'
                               ⊢ ∀ (x : R), Eq ((f.toRingHom.comp g.toRingHom) ((algebraMap R P.Ring) x)) ((a …
                             -/
  toRingHom_algebraMap := by simp [← IsScalarTower.algebraMap_apply]
                             /-
                               🎉 no goals
                             -/
                             /-
                               R : Type u
                               S : Type v
                               inst✝¹⁶ : CommRing R
                               inst✝¹⁵ : CommRing S
                               inst✝¹⁴ : Algebra R S
                               P : Algebra.Extension R S
                               R' : Type ?u.156843
                               S' : Type ?u.156846
                               inst✝¹³ : CommRing R'
                               inst✝¹² : CommRing S'
                               inst✝¹¹ : Algebra R' S'
                               P' : Algebra.Extension R' S'
                               R'' : Type ?u.157180
                               S'' : Type ?u.157183
                               inst✝¹⁰ : CommRing R''
                               inst✝⁹ : CommRing S''
                               inst✝⁸ : Algebra R'' S''
                               P'' : Algebra.Extension R'' S''
                               inst✝⁷ : Algebra R R'
                               inst✝⁶ : Algebra R' R''
                               inst✝⁵ : Algebra R R''
                               inst✝⁴ : Algebra S S'
                               inst✝³ : Algebra S' S''
                               inst✝² : Algebra S S''
                               inst✝¹ : IsScalarTower R R' R''
                               inst✝ : IsScalarTower S S' S''
                               f : P'.Hom P''
                               g : P.Hom P'
                               ⊢ ∀ (x : P.Ring), Eq ((algebraMap P''.Ring S'') ((f.toRingHom.comp g.toRingHom …
                             -/
  algebraMap_toRingHom := by simp [← IsScalarTower.algebraMap_apply]
                             /-
                               🎉 no goals
                             -/


@[simp]
                                                               /-
                                                                 R : Type u
                                                                 S : Type v
                                                                 inst✝⁷ : CommRing R
                                                                 inst✝⁶ : CommRing S
                                                                 inst✝⁵ : Algebra R S
                                                                 P : Algebra.Extension R S
                                                                 R' : Type u_1
                                                                 S' : Type u_2
                                                                 inst✝⁴ : CommRing R'
                                                                 inst✝³ : CommRing S'
                                                                 inst✝² : Algebra R' S'
                                                                 P' : Algebra.Extension R' S'
                                                                 inst✝¹ : Algebra R R'
                                                                 inst✝ : Algebra S S'
                                                                 f : P.Hom P'
                                                                 ⊢ Eq (f.comp (Algebra.Extension.Hom.id P)) f
                                                               -/
lemma Hom.comp_id (f : Hom P P') : f.comp (Hom.id P) = f := by ext; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
lemma Hom.id_comp (f : Hom P P') : (Hom.id P').comp f = f := by
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u_1
    S' : Type u_2
    inst✝⁴ : CommRing R'
    inst✝³ : CommRing S'
    inst✝² : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹ : Algebra R R'
    inst✝ : Algebra S S'
    f : P.Hom P'
    ⊢ Eq ((Algebra.Extension.Hom.id P').comp f) f
  -/
  ext; simp [Hom.id, aeval_X_left]
       /-
         🎉 no goals
       -/


/-- The kernel of an extension. -/
abbrev ker : Ideal P.Ring := RingHom.ker (algebraMap P.Ring S)


/-- The cotangent space of an extension.
This is a type synonym so that `P.Ring` can act on it through the action of `S` without creating
a diamond. -/
def Cotangent : Type _ := P.ker.Cotangent


noncomputable
instance : AddCommGroup P.Cotangent := inferInstanceAs (AddCommGroup P.ker.Cotangent)


/-- The identity map `P.ker.Cotangent → P.Cotangent` into the type synonym. -/
def Cotangent.of (x : P.ker.Cotangent) : P.Cotangent := x


/-- The identity map `P.Cotangent → P.ker.Cotangent` from the type synonym. -/
def Cotangent.val (x : P.Cotangent) : P.ker.Cotangent := x


@[ext]
lemma Cotangent.ext {x y : P.Cotangent} (e : x.val = y.val) : x = y := e


@[simp] lemma val_add : (x + y).val = x.val + y.val := rfl

@[simp] lemma val_zero : (0 : P.Cotangent).val = 0 := rfl

@[simp] lemma of_add : of (w + z) = of w + of z := rfl

@[simp] lemma of_zero : (of 0 : P.Cotangent) = 0 := rfl

@[simp] lemma of_val : of x.val = x := rfl

@[simp] lemma val_of : (of w).val = w := rfl

@[simp] lemma val_sub : (x - y).val = x.val - y.val := rfl


lemma Cotangent.smul_eq_zero_of_mem (p : P.Ring) (hp : p ∈ P.ker) (m : P.ker.Cotangent) :
    p • m = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    p : P.Ring
    hp : Membership.mem P.ker p
    m : P.ker.Cotangent
    ⊢ Eq (HSMul.hSMul p m) 0
  -/
  obtain ⟨x, rfl⟩ := Ideal.toCotangent_surjective _ m
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    p : P.Ring
    hp : Membership.mem P.ker p
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Eq (HSMul.hSMul p (P.ker.toCotangent x)) 0
  -/
  rw [← map_smul, Ideal.toCotangent_eq_zero, Submodule.coe_smul, smul_eq_mul, pow_two]
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    p : P.Ring
    hp : Membership.mem P.ker p
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Membership.mem (HMul.hMul P.ker P.ker) (HMul.hMul p ↑x)
  -/
  exact Ideal.mul_mem_mul hp x.2
  /-
    🎉 no goals
  -/


attribute [local simp] RingHom.mem_ker


noncomputable
instance Cotangent.module : Module S P.Cotangent where
  smul := fun r s ↦ .of (P.σ r • s.val)
  smul_zero := fun r ↦ ext (smul_zero (P.σ r))
  smul_add := fun r x y ↦ ext (smul_add (P.σ r) x.val y.val)
  add_smul := fun r s x ↦ by
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      r s : S
      x : P.Cotangent
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
    -/
    have := smul_eq_zero_of_mem (P.σ (r + s) - (P.σ r + P.σ s) : P.Ring) (by simp ) x
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      r s : S
      x : P.Cotangent
      this : Eq (HSMul.hSMul (HSub.hSub (P.σ (HAdd.hAdd r s)) (HAdd.hAdd (P.σ r) (P. …
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
    -/
    simpa only [sub_smul, add_smul, sub_eq_zero]
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      x : P.Cotangent
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      x : P.Cotangent
      this : Eq (HSMul.hSMul (HSub.hSub (P.σ 1) 1) x) 0
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
                                                                /-
                                                                  R : Type u
                                                                  S : Type v
                                                                  inst✝⁸ : CommRing R
                                                                  inst✝⁷ : CommRing S
                                                                  inst✝⁶ : Algebra R S
                                                                  P : Algebra.Extension R S
                                                                  R' : Type ?u.204330
                                                                  S' : Type ?u.204333
                                                                  inst✝⁵ : CommRing R'
                                                                  inst✝⁴ : CommRing S'
                                                                  inst✝³ : Algebra R' S'
                                                                  P' : Algebra.Extension R' S'
                                                                  R'' : Type ?u.204667
                                                                  S'' : Type ?u.204670
                                                                  inst✝² : CommRing R''
                                                                  inst✝¹ : CommRing S''
                                                                  inst✝ : Algebra R'' S''
                                                                  P'' : Algebra.Extension R'' S''
                                                                  x : P.Cotangent
                                                                  ⊢ Membership.mem P.ker (P.σ 0)
                                                                -/
    /-
      🎉 no goals
    -/
  zero_smul := fun x ↦ smul_eq_zero_of_mem (P.σ 0 : P.Ring) (by simp) x
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      r s : S
      x : P.Cotangent
      ⊢ Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.204330
      S' : Type ?u.204333
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.204667
      S'' : Type ?u.204670
      inst✝² : CommRing R''
      inst✝¹ : CommRing S''
      inst✝ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      r s : S
      x : P.Cotangent
      this : Eq (HSMul.hSMul (HSub.hSub (P.σ (HMul.hMul r s)) (HMul.hMul (P.σ r) (P. …
      ⊢ Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
  one_smul := fun x ↦ by
    /-
      🎉 no goals
    -/
    have := smul_eq_zero_of_mem (P.σ 1 - 1 : P.Ring) (by simp) x
    simpa [sub_eq_zero, sub_smul]
  mul_smul := fun r s x ↦ by
    have := smul_eq_zero_of_mem (P.σ (r * s) - (P.σ r * P.σ s) : P.Ring) (by simp) x
    simpa only [sub_smul, mul_smul, sub_eq_zero] using this


noncomputable
instance {R₀} [CommRing R₀] [Algebra R₀ S] : Module R₀ P.Cotangent :=
  Module.compHom P.Cotangent (algebraMap R₀ S)


instance {R₁ R₂} [CommRing R₁] [CommRing R₂] [Algebra R₁ S] [Algebra R₂ S] [Algebra R₁ R₂]
    [IsScalarTower R₁ R₂ S] :
  IsScalarTower R₁ R₂ P.Cotangent := by
  /-
    R : Type u
    S : Type v
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    P : Algebra.Extension R S
    R' : Type ?u.233885
    S' : Type ?u.233888
    inst✝¹¹ : CommRing R'
    inst✝¹⁰ : CommRing S'
    inst✝⁹ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type ?u.234222
    S'' : Type ?u.234225
    inst✝⁸ : CommRing R''
    inst✝⁷ : CommRing S''
    inst✝⁶ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁵ : CommRing R₁
    inst✝⁴ : CommRing R₂
    inst✝³ : Algebra R₁ S
    inst✝² : Algebra R₂ S
    inst✝¹ : Algebra R₁ R₂
    inst✝ : IsScalarTower R₁ R₂ S
    ⊢ IsScalarTower R₁ R₂ P.Cotangent
  -/
  constructor
  /-
    case smul_assoc
    R : Type u
    S : Type v
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    P : Algebra.Extension R S
    R' : Type ?u.233885
    S' : Type ?u.233888
    inst✝¹¹ : CommRing R'
    inst✝¹⁰ : CommRing S'
    inst✝⁹ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type ?u.234222
    S'' : Type ?u.234225
    inst✝⁸ : CommRing R''
    inst✝⁷ : CommRing S''
    inst✝⁶ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁵ : CommRing R₁
    inst✝⁴ : CommRing R₂
    inst✝³ : Algebra R₁ S
    inst✝² : Algebra R₂ S
    inst✝¹ : Algebra R₁ R₂
    inst✝ : IsScalarTower R₁ R₂ S
    ⊢ ∀ (x : R₁) (y : R₂) (z : P.Cotangent), Eq (HSMul.hSMul (HSMul.hSMul x y) z)  …
  -/
  intros r s m
  /-
    case smul_assoc
    R : Type u
    S : Type v
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    P : Algebra.Extension R S
    R' : Type ?u.233885
    S' : Type ?u.233888
    inst✝¹¹ : CommRing R'
    inst✝¹⁰ : CommRing S'
    inst✝⁹ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type ?u.234222
    S'' : Type ?u.234225
    inst✝⁸ : CommRing R''
    inst✝⁷ : CommRing S''
    inst✝⁶ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁵ : CommRing R₁
    inst✝⁴ : CommRing R₂
    inst✝³ : Algebra R₁ S
    inst✝² : Algebra R₂ S
    inst✝¹ : Algebra R₁ R₂
    inst✝ : IsScalarTower R₁ R₂ S
    r : R₁
    s : R₂
    m : P.Cotangent
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) m) (HSMul.hSMul r (HSMul.hSMul s m))
  -/
  show algebraMap R₂ S (r • s) • m = (algebraMap _ S r) • (algebraMap _ S s) • m
  /-
    case smul_assoc
    R : Type u
    S : Type v
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    P : Algebra.Extension R S
    R' : Type ?u.233885
    S' : Type ?u.233888
    inst✝¹¹ : CommRing R'
    inst✝¹⁰ : CommRing S'
    inst✝⁹ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type ?u.234222
    S'' : Type ?u.234225
    inst✝⁸ : CommRing R''
    inst✝⁷ : CommRing S''
    inst✝⁶ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁵ : CommRing R₁
    inst✝⁴ : CommRing R₂
    inst✝³ : Algebra R₁ S
    inst✝² : Algebra R₂ S
    inst✝¹ : Algebra R₁ R₂
    inst✝ : IsScalarTower R₁ R₂ S
    r : R₁
    s : R₂
    m : P.Cotangent
    ⊢ Eq (HSMul.hSMul ((algebraMap R₂ S) (HSMul.hSMul r s)) m) (HSMul.hSMul ((alge …
  -/
  rw [Algebra.smul_def, map_mul, mul_smul, ← IsScalarTower.algebraMap_apply]
  /-
    🎉 no goals
  -/


/-- The action of `R₀` on `P.Cotangent` for an extension `P → S`, if `S` is an `R₀` algebra. -/
lemma Cotangent.val_smul''' {R₀} [CommRing R₀] [Algebra R₀ S] (r : R₀) (x : P.Cotangent) :
    (r • x).val = P.σ (algebraMap R₀ S r) • x.val := rfl


/-- The action of `S` on `P.Cotangent` for an extension `P → S`. -/
@[simp]
lemma Cotangent.val_smul (r : S) (x : P.Cotangent) : (r • x).val = P.σ r • x.val := rfl


/-- The action of `P` on `P.Cotangent` for an extension `P → S`. -/
@[simp]
lemma Cotangent.val_smul' (r : P.Ring) (x : P.Cotangent) : (r • x).val = r • x.val := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    r : P.Ring
    x : P.Cotangent
    ⊢ Eq (HSMul.hSMul r x).val (HSMul.hSMul r x.val)
  -/
  rw [val_smul''', ← sub_eq_zero, ← sub_smul]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    r : P.Ring
    x : P.Cotangent
    ⊢ Eq (HSMul.hSMul (HSub.hSub (P.σ ((algebraMap P.Ring S) r)) r) x.val) 0
  -/
  exact Cotangent.smul_eq_zero_of_mem _ (by simp) _
  /-
    🎉 no goals
  -/


/-- The action of `R` on `P.Cotangent` for an `R`-extension `P → S`. -/
@[simp]
lemma Cotangent.val_smul'' (r : R) (x : P.Cotangent) : (r • x).val = r • x.val := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    r : R
    x : P.Cotangent
    ⊢ Eq (HSMul.hSMul r x).val (HSMul.hSMul r x.val)
  -/
  rw [← algebraMap_smul P.Ring, val_smul', algebraMap_smul]
  /-
    🎉 no goals
  -/


/-- The quotient map from the kernel of `P → S` onto the cotangent space. -/
def Cotangent.mk : P.ker →ₗ[P.Ring] P.Cotangent where
  toFun x := .of (Ideal.toCotangent _ x)
                     /-
                       R : Type u
                       S : Type v
                       inst✝⁸ : CommRing R
                       inst✝⁷ : CommRing S
                       inst✝⁶ : Algebra R S
                       P : Algebra.Extension R S
                       R' : Type ?u.260964
                       S' : Type ?u.260967
                       inst✝⁵ : CommRing R'
                       inst✝⁴ : CommRing S'
                       inst✝³ : Algebra R' S'
                       P' : Algebra.Extension R' S'
                       R'' : Type ?u.261301
                       S'' : Type ?u.261304
                       inst✝² : CommRing R''
                       inst✝¹ : CommRing S''
                       inst✝ : Algebra R'' S''
                       P'' : Algebra.Extension R'' S''
                       x y : Subtype fun x => Membership.mem P.ker x
                       ⊢ Eq ((fun x => Algebra.Extension.Cotangent.of (P.ker.toCotangent x)) (HAdd.hA …
                     -/
  map_add' x y := by simp
                     /-
                       🎉 no goals
                     -/
                             /-
                               R : Type u
                               S : Type v
                               inst✝⁸ : CommRing R
                               inst✝⁷ : CommRing S
                               inst✝⁶ : Algebra R S
                               P : Algebra.Extension R S
                               R' : Type ?u.260964
                               S' : Type ?u.260967
                               inst✝⁵ : CommRing R'
                               inst✝⁴ : CommRing S'
                               inst✝³ : Algebra R' S'
                               P' : Algebra.Extension R' S'
                               R'' : Type ?u.261301
                               S'' : Type ?u.261304
                               inst✝² : CommRing R''
                               inst✝¹ : CommRing S''
                               inst✝ : Algebra R'' S''
                               P'' : Algebra.Extension R'' S''
                               x : P.Ring
                               y : Subtype fun x => Membership.mem P.ker x
                               ⊢ Eq ({ toFun := fun x => Algebra.Extension.Cotangent.of (P.ker.toCotangent x) …
                             -/
  map_smul' x y := ext <| by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma Cotangent.val_mk (x : P.ker) : (mk x).val = Ideal.toCotangent _ x := rfl


lemma Cotangent.mk_surjective : Function.Surjective (mk (P := P)) :=
  fun x ↦ Ideal.toCotangent_surjective P.ker x.val


/-- A hom between two extensions induces a map between cotangent spaces. -/
noncomputable
def Cotangent.map (f : Hom P P') : P.Cotangent →ₗ[S] P'.Cotangent where
  toFun x := .of (Ideal.mapCotangent (R := R) _ _ f.toAlgHom
                   /-
                     R : Type u
                     S : Type v
                     inst✝¹⁶ : CommRing R
                     inst✝¹⁵ : CommRing S
                     inst✝¹⁴ : Algebra R S
                     P : Algebra.Extension R S
                     R' : Type ?u.277175
                     S' : Type ?u.277178
                     inst✝¹³ : CommRing R'
                     inst✝¹² : CommRing S'
                     inst✝¹¹ : Algebra R' S'
                     P' : Algebra.Extension R' S'
                     R'' : Type ?u.277512
                     S'' : Type ?u.277515
                     inst✝¹⁰ : CommRing R''
                     inst✝⁹ : CommRing S''
                     inst✝⁸ : Algebra R'' S''
                     P'' : Algebra.Extension R'' S''
                     inst✝⁷ : Algebra R R'
                     inst✝⁶ : Algebra R' R''
                     inst✝⁵ : Algebra R' S''
                     inst✝⁴ : Algebra S S'
                     inst✝³ : Algebra S' S''
                     inst✝² : Algebra S S''
                     inst✝¹ : Algebra R S'
                     inst✝ : IsScalarTower R R' S'
                     f : P.Hom P'
                     x✝ : P.Cotangent
                     x : P.Ring
                     hx : Membership.mem P.ker x
                     ⊢ Membership.mem (Ideal.comap f.toAlgHom P'.ker) x
                   -/
    (fun x hx ↦ by simpa using RingHom.congr_arg (algebraMap S S') hx) x.val)
                   /-
                     🎉 no goals
                   -/
  map_add' x y := ext (map_add _ x.val y.val)
  map_smul' r x := by
    /-
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      r : S
      x : P.Cotangent
      ⊢ Eq ({ toFun := fun x => Algebra.Extension.Cotangent.of ((P.ker.mapCotangent  …
    -/
    ext
    /-
      case e
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      r : S
      x : P.Cotangent
      ⊢ Eq ({ toFun := fun x => Algebra.Extension.Cotangent.of ((P.ker.mapCotangent  …
    -/
    obtain ⟨x, rfl⟩ := Cotangent.mk_surjective x
    /-
      case e.intro
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      r : S
      x : Subtype fun x => Membership.mem P.ker x
      ⊢ Eq ({ toFun := fun x => Algebra.Extension.Cotangent.of ((P.ker.mapCotangent  …
    -/
    obtain ⟨r, rfl⟩ := P.algebraMap_surjective r
    simp only [algebraMap_smul, val_smul', val_mk, val_of, Ideal.mapCotangent_toCotangent,
      RingHomCompTriple.comp_apply, ← (Ideal.toCotangent _).map_smul]
    conv_rhs => rw [← algebraMap_smul S', ← f.algebraMap_toRingHom, algebraMap_smul, val_smul',
      val_of, ← (Ideal.toCotangent _).map_smul]
    /-
      case e.intro.intro
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      x : Subtype fun x => Membership.mem P.ker x
      r : P.Ring
      ⊢ Eq (P'.ker.toCotangent ⟨f.toAlgHom ↑(HSMul.hSMul r x), ⋯⟩) (P'.ker.toCotange …
    -/
    congr 1
    /-
      case e.intro.intro.h.e_6.h
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      x : Subtype fun x => Membership.mem P.ker x
      r : P.Ring
      ⊢ Eq ⟨f.toAlgHom ↑(HSMul.hSMul r x), ⋯⟩ (HSMul.hSMul (f.toRingHom r) ⟨f.toAlgH …
    -/
    ext1
    /-
      case e.intro.intro.h.e_6.h.a
      R : Type u
      S : Type v
      inst✝¹⁶ : CommRing R
      inst✝¹⁵ : CommRing S
      inst✝¹⁴ : Algebra R S
      P : Algebra.Extension R S
      R' : Type ?u.277175
      S' : Type ?u.277178
      inst✝¹³ : CommRing R'
      inst✝¹² : CommRing S'
      inst✝¹¹ : Algebra R' S'
      P' : Algebra.Extension R' S'
      R'' : Type ?u.277512
      S'' : Type ?u.277515
      inst✝¹⁰ : CommRing R''
      inst✝⁹ : CommRing S''
      inst✝⁸ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra R' S''
      inst✝⁴ : Algebra S S'
      inst✝³ : Algebra S' S''
      inst✝² : Algebra S S''
      inst✝¹ : Algebra R S'
      inst✝ : IsScalarTower R R' S'
      f : P.Hom P'
      x : Subtype fun x => Membership.mem P.ker x
      r : P.Ring
      ⊢ Eq ↑⟨f.toAlgHom ↑(HSMul.hSMul r x), ⋯⟩ ↑(HSMul.hSMul (f.toRingHom r) ⟨f.toAl …
    -/
    simp only [SetLike.val_smul, smul_eq_mul, map_mul, Hom.toAlgHom_apply]
    /-
      🎉 no goals
    -/


@[simp]
lemma Cotangent.map_mk (f : Hom P P') (x) :
    Cotangent.map f (.mk x) =
                            /-
                              R : Type u
                              S : Type v
                              inst✝¹⁶ : CommRing R
                              inst✝¹⁵ : CommRing S
                              inst✝¹⁴ : Algebra R S
                              P : Algebra.Extension R S
                              R' : Type ?u.302519
                              S' : Type ?u.302522
                              inst✝¹³ : CommRing R'
                              inst✝¹² : CommRing S'
                              inst✝¹¹ : Algebra R' S'
                              P' : Algebra.Extension R' S'
                              R'' : Type ?u.302856
                              S'' : Type ?u.302859
                              inst✝¹⁰ : CommRing R''
                              inst✝⁹ : CommRing S''
                              inst✝⁸ : Algebra R'' S''
                              P'' : Algebra.Extension R'' S''
                              inst✝⁷ : Algebra R R'
                              inst✝⁶ : Algebra R' R''
                              inst✝⁵ : Algebra R' S''
                              inst✝⁴ : Algebra S S'
                              inst✝³ : Algebra S' S''
                              inst✝² : Algebra S S''
                              inst✝¹ : Algebra R S'
                              inst✝ : IsScalarTower R R' S'
                              f : P.Hom P'
                              x : Subtype fun x => Membership.mem P.ker x
                              ⊢ Membership.mem P'.ker (f.toAlgHom ↑x)
                            -/
      .mk ⟨f.toAlgHom x, by simpa [-map_aeval] using RingHom.congr_arg (algebraMap S S') x.2⟩ :=
                            /-
                              🎉 no goals
                            -/
  rfl


@[simp]
lemma Cotangent.map_id :
    Cotangent.map (.id P) = LinearMap.id := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    ⊢ Eq (Algebra.Extension.Cotangent.map (Algebra.Extension.Hom.id P)) LinearMap.id
  -/
  ext x
  /-
    case h.e
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    x : P.Cotangent
    ⊢ Eq ((Algebra.Extension.Cotangent.map (Algebra.Extension.Hom.id P)) x).val (L …
  -/
  obtain ⟨x, rfl⟩ := Cotangent.mk_surjective x
  simp only [map_mk, Hom.toAlgHom_id, AlgHom.coe_id, id_eq, Subtype.coe_eta, val_mk,
    LinearMap.id_coe]


lemma Cotangent.map_comp (f : Hom P P') (g : Hom P' P'') :
    Cotangent.map (g.comp f) = (map g).restrictScalars S ∘ₗ map f := by
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u_1
    S' : Type u_2
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type u_4
    S'' : Type u_5
    inst✝¹⁶ : CommRing R''
    inst✝¹⁵ : CommRing S''
    inst✝¹⁴ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹³ : Algebra R R'
    inst✝¹² : Algebra R' R''
    inst✝¹¹ : Algebra R' S''
    inst✝¹⁰ : Algebra S S'
    inst✝⁹ : Algebra S' S''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S'
    inst✝⁶ : IsScalarTower R R' S'
    inst✝⁵ : Algebra R R''
    inst✝⁴ : IsScalarTower R R' R''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : Algebra R S''
    inst✝¹ : IsScalarTower R R'' S''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    g : P'.Hom P''
    ⊢ Eq (Algebra.Extension.Cotangent.map (g.comp f)) ((↑S (Algebra.Extension.Cota …
  -/
  ext x
  /-
    case h.e
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u_1
    S' : Type u_2
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    R'' : Type u_4
    S'' : Type u_5
    inst✝¹⁶ : CommRing R''
    inst✝¹⁵ : CommRing S''
    inst✝¹⁴ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹³ : Algebra R R'
    inst✝¹² : Algebra R' R''
    inst✝¹¹ : Algebra R' S''
    inst✝¹⁰ : Algebra S S'
    inst✝⁹ : Algebra S' S''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S'
    inst✝⁶ : IsScalarTower R R' S'
    inst✝⁵ : Algebra R R''
    inst✝⁴ : IsScalarTower R R' R''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : Algebra R S''
    inst✝¹ : IsScalarTower R R'' S''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    g : P'.Hom P''
    x : P.Cotangent
    ⊢ Eq ((Algebra.Extension.Cotangent.map (g.comp f)) x).val (((↑S (Algebra.Exten …
  -/
  obtain ⟨x, rfl⟩ := Cotangent.mk_surjective x
  simp only [map_mk, Hom.toAlgHom_apply, Hom.comp_toRingHom, RingHom.coe_comp, Function.comp_apply,
    val_mk, LinearMap.coe_comp, LinearMap.coe_restrictScalars]


