/-- If a type carries an involutive star, then any star-closed subset does too. -/
instance instInvolutiveStar {S R : Type*} [InvolutiveStar R] [SetLike S R] [StarMemClass S R]
    (s : S) : InvolutiveStar s where
  star_involutive r := Subtype.ext <| star_star (r : R)


/-- In a star magma (i.e., a multiplication with an antimultiplicative involutive star
operation), any star-closed subset which is also closed under multiplication is itself a star
magma. -/
instance instStarMul {S R : Type*} [Mul R] [StarMul R] [SetLike S R]
    [MulMemClass S R] [StarMemClass S R] (s : S) : StarMul s where
  star_mul _ _ := Subtype.ext <| star_mul _ _


/-- In a `StarAddMonoid` (i.e., an additive monoid with an additive involutive star operation), any
star-closed subset which is also closed under addition and contains zero is itself a
`StarAddMonoid`. -/
instance instStarAddMonoid {S R : Type*} [AddMonoid R] [StarAddMonoid R] [SetLike S R]
    [AddSubmonoidClass S R] [StarMemClass S R] (s : S) : StarAddMonoid s where
  star_add _ _ := Subtype.ext <| star_add _ _


/-- In a star ring (i.e., a non-unital, non-associative, semiring with an additive,
antimultiplicative, involutive star operation), a star-closed non-unital subsemiring is itself a
star ring. -/
instance instStarRing {S R : Type*} [NonUnitalNonAssocSemiring R] [StarRing R] [SetLike S R]
    [NonUnitalSubsemiringClass S R] [StarMemClass S R] (s : S) : StarRing s :=
  { StarMemClass.instStarMul s, StarMemClass.instStarAddMonoid s with }


/-- In a star `R`-module (i.e., `star (r • m) = (star r) • m`) any star-closed subset which is also
closed under the scalar action by `R` is itself a star `R`-module. -/
instance instStarModule {S : Type*} (R : Type*) {M : Type*} [Star R] [Star M] [SMul R M]
    [StarModule R M] [SetLike S M] [SMulMemClass S R M] [StarMemClass S M] (s : S) :
    StarModule R s where
  star_smul _ _ := Subtype.ext <| star_smul _ _


/-- Embedding of a non-unital star subalgebra into the non-unital star algebra. -/
def subtype (s : S) : s →⋆ₙₐ[R] A :=
  { NonUnitalSubalgebraClass.subtype s with
    toFun := Subtype.val
    map_star' := fun _ => rfl }


@[simp]
theorem coeSubtype : (subtype s : s → A) = Subtype.val :=
  rfl


/-- A non-unital star subalgebra is a non-unital subalgebra which is closed under the `star`
operation. -/
structure NonUnitalStarSubalgebra (R : Type u) (A : Type v) [CommSemiring R]
    [NonUnitalNonAssocSemiring A] [Module R A] [Star A]
    extends NonUnitalSubalgebra R A : Type v where
  /-- The `carrier` of a `NonUnitalStarSubalgebra` is closed under the `star` operation. -/
  star_mem' : ∀ {a : A} (_ha : a ∈ carrier), star a ∈ carrier


instance instSetLike : SetLike (NonUnitalStarSubalgebra R A) A where
  coe {s} := s.carrier
                             /-
                               F : Type v'
                               R' : Type u'
                               R : Type u
                               A : Type v
                               B : Type w
                               C : Type w'
                               inst✝¹² : CommSemiring R
                               inst✝¹¹ : NonUnitalNonAssocSemiring A
                               inst✝¹⁰ : Module R A
                               inst✝⁹ : Star A
                               inst✝⁸ : NonUnitalNonAssocSemiring B
                               inst✝⁷ : Module R B
                               inst✝⁶ : Star B
                               inst✝⁵ : NonUnitalNonAssocSemiring C
                               inst✝⁴ : Module R C
                               inst✝³ : Star C
                               inst✝² : FunLike F A B
                               inst✝¹ : NonUnitalAlgHomClass F R A B
                               inst✝ : StarHomClass F A B
                               p q : NonUnitalStarSubalgebra R A
                               h : Eq (fun {s} => s.carrier) fun {s} => s.carrier
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance instNonUnitalSubsemiringClass :
    NonUnitalSubsemiringClass (NonUnitalStarSubalgebra R A) A where
  add_mem {s} := s.add_mem'
  mul_mem {s} := s.mul_mem'
  zero_mem {s} := s.zero_mem'


instance instSMulMemClass : SMulMemClass (NonUnitalStarSubalgebra R A) R A where
  smul_mem {s} := s.smul_mem'


instance instStarMemClass : StarMemClass (NonUnitalStarSubalgebra R A) A where
  star_mem {s} := s.star_mem'


instance instNonUnitalSubringClass {R : Type u} {A : Type v} [CommRing R] [NonUnitalNonAssocRing A]
    [Module R A] [Star A] : NonUnitalSubringClass (NonUnitalStarSubalgebra R A) A :=
  { NonUnitalStarSubalgebra.instNonUnitalSubsemiringClass with
    neg_mem := fun _S {x} hx => neg_one_smul R x ▸ SMulMemClass.smul_mem _ hx }


theorem mem_carrier {s : NonUnitalStarSubalgebra R A} {x : A} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[ext]
theorem ext {S T : NonUnitalStarSubalgebra R A} (h : ∀ x : A, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
theorem mem_toNonUnitalSubalgebra {S : NonUnitalStarSubalgebra R A} {x} :
    x ∈ S.toNonUnitalSubalgebra ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalSubalgebra (S : NonUnitalStarSubalgebra R A) :
    (↑S.toNonUnitalSubalgebra : Set A) = S :=
  rfl


theorem toNonUnitalSubalgebra_injective :
    Function.Injective
      (toNonUnitalSubalgebra : NonUnitalStarSubalgebra R A → NonUnitalSubalgebra R A) :=
  fun S T h =>
                  /-
                    R : Type u
                    A : Type v
                    inst✝³ : CommSemiring R
                    inst✝² : NonUnitalNonAssocSemiring A
                    inst✝¹ : Module R A
                    inst✝ : Star A
                    S T : NonUnitalStarSubalgebra R A
                    h : Eq S.toNonUnitalSubalgebra T.toNonUnitalSubalgebra
                    x : A
                    ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                  -/
  ext fun x => by rw [← mem_toNonUnitalSubalgebra, ← mem_toNonUnitalSubalgebra, h]
                  /-
                    🎉 no goals
                  -/


theorem toNonUnitalSubalgebra_inj {S U : NonUnitalStarSubalgebra R A} :
    S.toNonUnitalSubalgebra = U.toNonUnitalSubalgebra ↔ S = U :=
  toNonUnitalSubalgebra_injective.eq_iff


theorem toNonUnitalSubalgebra_le_iff {S₁ S₂ : NonUnitalStarSubalgebra R A} :
    S₁.toNonUnitalSubalgebra ≤ S₂.toNonUnitalSubalgebra ↔ S₁ ≤ S₂ :=
  Iff.rfl


/-- Copy of a non-unital star subalgebra with a new `carrier` equal to the old one.
Useful to fix definitional equalities. -/
protected def copy (S : NonUnitalStarSubalgebra R A) (s : Set A) (hs : s = ↑S) :
    NonUnitalStarSubalgebra R A :=
  { S.toNonUnitalSubalgebra.copy s hs with
    star_mem' := @fun x (hx : x ∈ s) => by
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : NonUnitalNonAssocSemiring A
        inst✝¹⁰ : Module R A
        inst✝⁹ : Star A
        inst✝⁸ : NonUnitalNonAssocSemiring B
        inst✝⁷ : Module R B
        inst✝⁶ : Star B
        inst✝⁵ : NonUnitalNonAssocSemiring C
        inst✝⁴ : Module R C
        inst✝³ : Star C
        inst✝² : FunLike F A B
        inst✝¹ : NonUnitalAlgHomClass F R A B
        inst✝ : StarHomClass F A B
        S : NonUnitalStarSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        x : A
        hx : Membership.mem s x
        ⊢ Membership.mem __src✝.carrier (Star.star x)
      -/
      show star x ∈ s
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : NonUnitalNonAssocSemiring A
        inst✝¹⁰ : Module R A
        inst✝⁹ : Star A
        inst✝⁸ : NonUnitalNonAssocSemiring B
        inst✝⁷ : Module R B
        inst✝⁶ : Star B
        inst✝⁵ : NonUnitalNonAssocSemiring C
        inst✝⁴ : Module R C
        inst✝³ : Star C
        inst✝² : FunLike F A B
        inst✝¹ : NonUnitalAlgHomClass F R A B
        inst✝ : StarHomClass F A B
        S : NonUnitalStarSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        x : A
        hx : Membership.mem s x
        ⊢ Membership.mem s (Star.star x)
      -/
      rw [hs] at hx ⊢
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : NonUnitalNonAssocSemiring A
        inst✝¹⁰ : Module R A
        inst✝⁹ : Star A
        inst✝⁸ : NonUnitalNonAssocSemiring B
        inst✝⁷ : Module R B
        inst✝⁶ : Star B
        inst✝⁵ : NonUnitalNonAssocSemiring C
        inst✝⁴ : Module R C
        inst✝³ : Star C
        inst✝² : FunLike F A B
        inst✝¹ : NonUnitalAlgHomClass F R A B
        inst✝ : StarHomClass F A B
        S : NonUnitalStarSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        x : A
        hx : Membership.mem (↑S) x
        ⊢ Membership.mem (↑S) (Star.star x)
      -/
      exact S.star_mem' hx }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_copy (S : NonUnitalStarSubalgebra R A) (s : Set A) (hs : s = ↑S) :
    (S.copy s hs : Set A) = s :=
  rfl


theorem copy_eq (S : NonUnitalStarSubalgebra R A) (s : Set A) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


/-- A non-unital star subalgebra over a ring is also a `Subring`. -/
def toNonUnitalSubring {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] (S : NonUnitalStarSubalgebra R A) : NonUnitalSubring A where
  toNonUnitalSubsemiring := S.toNonUnitalSubsemiring
  neg_mem' := neg_mem (s := S)


@[simp]
theorem mem_toNonUnitalSubring {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] {S : NonUnitalStarSubalgebra R A} {x} : x ∈ S.toNonUnitalSubring ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalSubring {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] (S : NonUnitalStarSubalgebra R A) : (↑S.toNonUnitalSubring : Set A) = S :=
  rfl


theorem toNonUnitalSubring_injective {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A]
    [Module R A] [Star A] :
    Function.Injective (toNonUnitalSubring : NonUnitalStarSubalgebra R A → NonUnitalSubring A) :=
                               /-
                                 R : Type u
                                 A : Type v
                                 inst✝³ : CommRing R
                                 inst✝² : NonUnitalRing A
                                 inst✝¹ : Module R A
                                 inst✝ : Star A
                                 S T : NonUnitalStarSubalgebra R A
                                 h : Eq S.toNonUnitalSubring T.toNonUnitalSubring
                                 x : A
                                 ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                               -/
  fun S T h => ext fun x => by rw [← mem_toNonUnitalSubring, ← mem_toNonUnitalSubring, h]
                               /-
                                 🎉 no goals
                               -/


theorem toNonUnitalSubring_inj {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] {S U : NonUnitalStarSubalgebra R A} :
    S.toNonUnitalSubring = U.toNonUnitalSubring ↔ S = U :=
  toNonUnitalSubring_injective.eq_iff


instance instInhabited : Inhabited S :=
  ⟨(0 : S.toNonUnitalSubalgebra)⟩


instance toNonUnitalSemiring {R A} [CommSemiring R] [NonUnitalSemiring A] [Module R A] [Star A]
    (S : NonUnitalStarSubalgebra R A) : NonUnitalSemiring S :=
  inferInstance


instance toNonUnitalCommSemiring {R A} [CommSemiring R] [NonUnitalCommSemiring A] [Module R A]
    [Star A] (S : NonUnitalStarSubalgebra R A) : NonUnitalCommSemiring S :=
  inferInstance


instance toNonUnitalRing {R A} [CommRing R] [NonUnitalRing A] [Module R A] [Star A]
    (S : NonUnitalStarSubalgebra R A) : NonUnitalRing S :=
  inferInstance


instance toNonUnitalCommRing {R A} [CommRing R] [NonUnitalCommRing A] [Module R A] [Star A]
    (S : NonUnitalStarSubalgebra R A) : NonUnitalCommRing S :=
  inferInstance

/-- The forgetful map from `NonUnitalStarSubalgebra` to `NonUnitalSubalgebra` as an
`OrderEmbedding` -/
def toNonUnitalSubalgebra' : NonUnitalStarSubalgebra R A ↪o NonUnitalSubalgebra R A where
  toEmbedding :=
    { toFun := fun S => S.toNonUnitalSubalgebra
                                     /-
                                       F : Type v'
                                       R' : Type u'
                                       R : Type u
                                       A : Type v
                                       B : Type w
                                       C : Type w'
                                       inst✝¹² : CommSemiring R
                                       inst✝¹¹ : NonUnitalNonAssocSemiring A
                                       inst✝¹⁰ : Module R A
                                       inst✝⁹ : Star A
                                       inst✝⁸ : NonUnitalNonAssocSemiring B
                                       inst✝⁷ : Module R B
                                       inst✝⁶ : Star B
                                       inst✝⁵ : NonUnitalNonAssocSemiring C
                                       inst✝⁴ : Module R C
                                       inst✝³ : Star C
                                       inst✝² : FunLike F A B
                                       inst✝¹ : NonUnitalAlgHomClass F R A B
                                       inst✝ : StarHomClass F A B
                                       S✝ S T : NonUnitalStarSubalgebra R A
                                       h : Eq ((fun S => S.toNonUnitalSubalgebra) S) ((fun S => S.toNonUnitalSubalgeb …
                                       ⊢ ∀ (x : A), Iff (Membership.mem S x) (Membership.mem T x)
                                     -/
      inj' := fun S T h => ext <| by apply SetLike.ext_iff.1 h }
                                     /-
                                       🎉 no goals
                                     -/
  map_rel_iff' := SetLike.coe_subset_coe.symm.trans SetLike.coe_subset_coe


instance module' [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] : Module R' S :=
  SMulMemClass.toModule' _ R' R A S


instance instModule : Module R S :=
  S.module'


instance instIsScalarTower' [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] :
    IsScalarTower R' R S :=
  S.toNonUnitalSubalgebra.instIsScalarTower'


instance instIsScalarTower [IsScalarTower R A A] : IsScalarTower R S S where
  smul_assoc r x y := Subtype.ext <| smul_assoc r (x : A) (y : A)


instance instSMulCommClass' [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A]
    [SMulCommClass R' R A] : SMulCommClass R' R S where
  smul_comm r' r s := Subtype.ext <| smul_comm r' r (s : A)


instance instSMulCommClass [SMulCommClass R A A] : SMulCommClass R S S where
  smul_comm r x y := Subtype.ext <| smul_comm r (x : A) (y : A)


instance noZeroSMulDivisors_bot [NoZeroSMulDivisors R A] : NoZeroSMulDivisors R S :=
  ⟨fun {c x} h =>
    have : c = 0 ∨ (x : A) = 0 := eq_zero_or_eq_zero_of_smul_eq_zero (congr_arg ((↑) : S → A) h)
    this.imp_right (@Subtype.ext_iff _ _ x 0).mpr⟩


protected theorem coe_add (x y : S) : (↑(x + y) : A) = ↑x + ↑y :=
  rfl


protected theorem coe_mul (x y : S) : (↑(x * y) : A) = ↑x * ↑y :=
  rfl


protected theorem coe_zero : ((0 : S) : A) = 0 :=
  rfl


protected theorem coe_neg {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] {S : NonUnitalStarSubalgebra R A} (x : S) : (↑(-x) : A) = -↑x :=
  rfl


protected theorem coe_sub {R : Type u} {A : Type v} [CommRing R] [NonUnitalRing A] [Module R A]
    [Star A] {S : NonUnitalStarSubalgebra R A} (x y : S) : (↑(x - y) : A) = ↑x - ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_smul [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] (r : R') (x : S) :
    ↑(r • x) = r • (x : A) :=
  rfl


protected theorem coe_eq_zero {x : S} : (x : A) = 0 ↔ x = 0 :=
  ZeroMemClass.coe_eq_zero


@[simp]
theorem toNonUnitalSubalgebra_subtype :
    NonUnitalSubalgebraClass.subtype S = NonUnitalStarSubalgebraClass.subtype S :=
  rfl


@[simp]
theorem toSubring_subtype {R A : Type*} [CommRing R] [NonUnitalRing A] [Module R A] [Star A]
    (S : NonUnitalStarSubalgebra R A) :
    NonUnitalSubringClass.subtype S = NonUnitalStarSubalgebraClass.subtype S :=
  rfl


/-- Transport a non-unital star subalgebra via a non-unital star algebra homomorphism. -/
def map (f : F) (S : NonUnitalStarSubalgebra R A) : NonUnitalStarSubalgebra R B where
  toNonUnitalSubalgebra := S.toNonUnitalSubalgebra.map (f : A →ₙₐ[R] B)
                  /-
                    F : Type v'
                    R' : Type u'
                    R : Type u
                    A : Type v
                    B : Type w
                    C : Type w'
                    inst✝¹² : CommSemiring R
                    inst✝¹¹ : NonUnitalNonAssocSemiring A
                    inst✝¹⁰ : Module R A
                    inst✝⁹ : Star A
                    inst✝⁸ : NonUnitalNonAssocSemiring B
                    inst✝⁷ : Module R B
                    inst✝⁶ : Star B
                    inst✝⁵ : NonUnitalNonAssocSemiring C
                    inst✝⁴ : Module R C
                    inst✝³ : Star C
                    inst✝² : FunLike F A B
                    inst✝¹ : NonUnitalAlgHomClass F R A B
                    inst✝ : StarHomClass F A B
                    S✝ : NonUnitalStarSubalgebra R A
                    f : F
                    S : NonUnitalStarSubalgebra R A
                    ⊢ ∀ {a : B}, Membership.mem (NonUnitalSubalgebra.map (NonUnitalAlgHomClass.toN …
                  -/
  star_mem' := by rintro _ ⟨a, ha, rfl⟩; exact ⟨star a, star_mem (s := S) ha, map_star f a⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem map_mono {S₁ S₂ : NonUnitalStarSubalgebra R A} {f : F} :
    S₁ ≤ S₂ → (map f S₁ : NonUnitalStarSubalgebra R B) ≤ map f S₂ :=
  Set.image_subset f


theorem map_injective {f : F} (hf : Function.Injective f) :
    Function.Injective (map f : NonUnitalStarSubalgebra R A → NonUnitalStarSubalgebra R B) :=
  fun _S₁ _S₂ ih =>
  ext <| Set.ext_iff.1 <| Set.image_injective.2 hf <| Set.ext <| SetLike.ext_iff.mp ih


@[simp]
theorem map_id (S : NonUnitalStarSubalgebra R A) : map (NonUnitalStarAlgHom.id R A) S = S :=
  SetLike.coe_injective <| Set.image_id _


theorem map_map (S : NonUnitalStarSubalgebra R A) (g : B →⋆ₙₐ[R] C) (f : A →⋆ₙₐ[R] B) :
    (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image _ _ _


@[simp]
theorem mem_map {S : NonUnitalStarSubalgebra R A} {f : F} {y : B} :
    y ∈ map f S ↔ ∃ x ∈ S, f x = y :=
  NonUnitalSubalgebra.mem_map


theorem map_toNonUnitalSubalgebra {S : NonUnitalStarSubalgebra R A} {f : F} :
    (map f S : NonUnitalStarSubalgebra R B).toNonUnitalSubalgebra =
      NonUnitalSubalgebra.map f S.toNonUnitalSubalgebra :=
  SetLike.coe_injective rfl


@[simp]
theorem coe_map (S : NonUnitalStarSubalgebra R A) (f : F) : map f S = f '' S :=
  rfl


/-- Preimage of a non-unital star subalgebra under a non-unital star algebra homomorphism. -/
def comap (f : F) (S : NonUnitalStarSubalgebra R B) : NonUnitalStarSubalgebra R A where
  toNonUnitalSubalgebra := S.toNonUnitalSubalgebra.comap f
  star_mem' := @fun a (ha : f a ∈ S) =>
    show f (star a) ∈ S from (map_star f a).symm ▸ star_mem (s := S) ha


theorem map_le {S : NonUnitalStarSubalgebra R A} {f : F} {U : NonUnitalStarSubalgebra R B} :
    map f S ≤ U ↔ S ≤ comap f U :=
  Set.image_subset_iff


theorem gc_map_comap (f : F) : GaloisConnection (map f) (comap f) :=
  fun _S _U => map_le


@[simp]
theorem mem_comap (S : NonUnitalStarSubalgebra R B) (f : F) (x : A) : x ∈ comap f S ↔ f x ∈ S :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_comap (S : NonUnitalStarSubalgebra R B) (f : F) : comap f S = f ⁻¹' (S : Set B) :=
  rfl


instance instNoZeroDivisors {R A : Type*} [CommSemiring R] [NonUnitalSemiring A] [NoZeroDivisors A]
    [Module R A] [Star A] (S : NonUnitalStarSubalgebra R A) : NoZeroDivisors S :=
  NonUnitalSubsemiringClass.noZeroDivisors S


/-- A non-unital subalgebra closed under `star` is a non-unital star subalgebra. -/
def toNonUnitalStarSubalgebra (h_star : ∀ x, x ∈ s → star x ∈ s) : NonUnitalStarSubalgebra R A :=
  { s with
    star_mem' := @h_star }


@[simp]
theorem mem_toNonUnitalStarSubalgebra {s : NonUnitalSubalgebra R A} {h_star} {x} :
    x ∈ s.toNonUnitalStarSubalgebra h_star ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalStarSubalgebra (s : NonUnitalSubalgebra R A) (h_star) :
    (s.toNonUnitalStarSubalgebra h_star : Set A) = s :=
  rfl


@[simp]
theorem toNonUnitalStarSubalgebra_toNonUnitalSubalgebra (s : NonUnitalSubalgebra R A) (h_star) :
    (s.toNonUnitalStarSubalgebra h_star).toNonUnitalSubalgebra = s :=
  SetLike.coe_injective rfl


@[simp]
theorem _root_.NonUnitalStarSubalgebra.toNonUnitalSubalgebra_toNonUnitalStarSubalgebra
    (S : NonUnitalStarSubalgebra R A) :
    (S.toNonUnitalSubalgebra.toNonUnitalStarSubalgebra fun _ => star_mem (s := S)) = S :=
  SetLike.coe_injective rfl


/-- Range of an `NonUnitalAlgHom` as a `NonUnitalStarSubalgebra`. -/
protected def range (φ : F) : NonUnitalStarSubalgebra R B where
  toNonUnitalSubalgebra := NonUnitalAlgHom.range (φ : A →ₙₐ[R] B)
                  /-
                    F : Type v'
                    R' : Type u'
                    R : Type u
                    A : Type v
                    B : Type w
                    C : Type w'
                    inst✝¹² : CommSemiring R
                    inst✝¹¹ : NonUnitalNonAssocSemiring A
                    inst✝¹⁰ : Module R A
                    inst✝⁹ : Star A
                    inst✝⁸ : NonUnitalNonAssocSemiring B
                    inst✝⁷ : Module R B
                    inst✝⁶ : Star B
                    inst✝⁵ : NonUnitalNonAssocSemiring C
                    inst✝⁴ : Module R C
                    inst✝³ : Star C
                    inst✝² : FunLike F A B
                    inst✝¹ : NonUnitalAlgHomClass F R A B
                    inst✝ : StarHomClass F A B
                    φ : F
                    ⊢ ∀ {a : B}, Membership.mem (NonUnitalAlgHom.range (NonUnitalAlgHomClass.toNon …
                  -/
  star_mem' := by rintro _ ⟨a, rfl⟩; exact ⟨star a, map_star φ a⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem mem_range (φ : F) {y : B} :
    y ∈ (NonUnitalStarAlgHom.range φ : NonUnitalStarSubalgebra R B) ↔ ∃ x : A, φ x = y :=
  NonUnitalRingHom.mem_srange


theorem mem_range_self (φ : F) (x : A) :
    φ x ∈ (NonUnitalStarAlgHom.range φ : NonUnitalStarSubalgebra R B) :=
  (NonUnitalAlgHom.mem_range φ).2 ⟨x, rfl⟩


@[simp]
theorem coe_range (φ : F) :
    ((NonUnitalStarAlgHom.range φ : NonUnitalStarSubalgebra R B) : Set B) = Set.range (φ : A → B) :=
     /-
       F : Type v'
       R : Type u
       A : Type v
       B : Type w
       inst✝⁹ : CommSemiring R
       inst✝⁸ : NonUnitalNonAssocSemiring A
       inst✝⁷ : Module R A
       inst✝⁶ : Star A
       inst✝⁵ : NonUnitalNonAssocSemiring B
       inst✝⁴ : Module R B
       inst✝³ : Star B
       inst✝² : FunLike F A B
       inst✝¹ : NonUnitalAlgHomClass F R A B
       inst✝ : StarHomClass F A B
       φ : F
       ⊢ Eq (↑(NonUnitalStarAlgHom.range φ)) (Set.range ⇑φ)
     -/
  by ext; rw [SetLike.mem_coe, mem_range]; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem range_comp (f : A →⋆ₙₐ[R] B) (g : B →⋆ₙₐ[R] C) :
    NonUnitalStarAlgHom.range (g.comp f) = (NonUnitalStarAlgHom.range f).map g :=
  SetLike.coe_injective (Set.range_comp g f)


theorem range_comp_le_range (f : A →⋆ₙₐ[R] B) (g : B →⋆ₙₐ[R] C) :
    NonUnitalStarAlgHom.range (g.comp f) ≤ NonUnitalStarAlgHom.range g :=
  SetLike.coe_mono (Set.range_comp_subset_range f g)


/-- Restrict the codomain of a non-unital star algebra homomorphism. -/
def codRestrict (f : F) (S : NonUnitalStarSubalgebra R B) (hf : ∀ x, f x ∈ S) : A →⋆ₙₐ[R] S where
  toNonUnitalAlgHom := NonUnitalAlgHom.codRestrict f S.toNonUnitalSubalgebra hf
  map_star' := fun a => Subtype.ext <| map_star f a


@[simp]
theorem subtype_comp_codRestrict (f : F) (S : NonUnitalStarSubalgebra R B) (hf : ∀ x : A, f x ∈ S) :
    (NonUnitalStarSubalgebraClass.subtype S).comp (NonUnitalStarAlgHom.codRestrict f S hf) = f :=
  NonUnitalStarAlgHom.ext fun _ => rfl


@[simp]
theorem coe_codRestrict (f : F) (S : NonUnitalStarSubalgebra R B) (hf : ∀ x, f x ∈ S) (x : A) :
    ↑(NonUnitalStarAlgHom.codRestrict f S hf x) = f x :=
  rfl


theorem injective_codRestrict (f : F) (S : NonUnitalStarSubalgebra R B) (hf : ∀ x : A, f x ∈ S) :
    Function.Injective (NonUnitalStarAlgHom.codRestrict f S hf) ↔ Function.Injective f :=
  ⟨fun H _x _y hxy => H <| Subtype.eq hxy, fun H _x _y hxy => H (congr_arg Subtype.val hxy : _)⟩


/-- Restrict the codomain of a non-unital star algebra homomorphism `f` to `f.range`.

This is the bundled version of `Set.rangeFactorization`. -/
abbrev rangeRestrict (f : F) :
    A →⋆ₙₐ[R] (NonUnitalStarAlgHom.range f : NonUnitalStarSubalgebra R B) :=
  NonUnitalStarAlgHom.codRestrict f (NonUnitalStarAlgHom.range f)
    (NonUnitalStarAlgHom.mem_range_self f)


/-- The equalizer of two non-unital star `R`-algebra homomorphisms -/
def equalizer (ϕ ψ : F) : NonUnitalStarSubalgebra R A where
  toNonUnitalSubalgebra := NonUnitalAlgHom.equalizer ϕ ψ
                                             /-
                                               F : Type v'
                                               R' : Type u'
                                               R : Type u
                                               A : Type v
                                               B : Type w
                                               C : Type w'
                                               inst✝¹² : CommSemiring R
                                               inst✝¹¹ : NonUnitalNonAssocSemiring A
                                               inst✝¹⁰ : Module R A
                                               inst✝⁹ : Star A
                                               inst✝⁸ : NonUnitalNonAssocSemiring B
                                               inst✝⁷ : Module R B
                                               inst✝⁶ : Star B
                                               inst✝⁵ : NonUnitalNonAssocSemiring C
                                               inst✝⁴ : Module R C
                                               inst✝³ : Star C
                                               inst✝² : FunLike F A B
                                               inst✝¹ : NonUnitalAlgHomClass F R A B
                                               inst✝ : StarHomClass F A B
                                               ϕ ψ : F
                                               x : A
                                               hx : Eq (ϕ x) (ψ x)
                                               ⊢ Membership.mem (NonUnitalAlgHom.equalizer ϕ ψ).carrier (Star.star x)
                                             -/
  star_mem' := @fun x (hx : ϕ x = ψ x) => by simp [map_star, hx]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem mem_equalizer (φ ψ : F) (x : A) :
    x ∈ NonUnitalStarAlgHom.equalizer φ ψ ↔ φ x = ψ x :=
  Iff.rfl


/-- Restrict a non-unital star algebra homomorphism with a left inverse to an algebra isomorphism
to its range.

This is a computable alternative to `StarAlgEquiv.ofInjective`. -/
def ofLeftInverse' {g : B → A} {f : F} (h : Function.LeftInverse g f) :
    A ≃⋆ₐ[R] NonUnitalStarAlgHom.range f :=
  { NonUnitalStarAlgHom.rangeRestrict f with
    toFun := NonUnitalStarAlgHom.rangeRestrict f
    invFun := g ∘ (NonUnitalStarSubalgebraClass.subtype <| NonUnitalStarAlgHom.range f)
    left_inv := h
    right_inv := fun x =>
      Subtype.ext <|
        let ⟨x', hx'⟩ := (NonUnitalStarAlgHom.mem_range f).mp x.prop
                            /-
                              F : Type v'
                              R' : Type u'
                              R : Type u
                              A : Type v
                              B : Type w
                              C : Type w'
                              inst✝¹² : CommSemiring R
                              inst✝¹¹ : NonUnitalSemiring A
                              inst✝¹⁰ : Module R A
                              inst✝⁹ : Star A
                              inst✝⁸ : NonUnitalSemiring B
                              inst✝⁷ : Module R B
                              inst✝⁶ : Star B
                              inst✝⁵ : NonUnitalSemiring C
                              inst✝⁴ : Module R C
                              inst✝³ : Star C
                              inst✝² : FunLike F A B
                              inst✝¹ : NonUnitalAlgHomClass F R A B
                              inst✝ : StarHomClass F A B
                              g : B → A
                              f : F
                              h : Function.LeftInverse g ⇑f
                              x : Subtype fun x => Membership.mem (NonUnitalStarAlgHom.range f) x
                              x' : A
                              hx' : Eq (f x') ↑x
                              ⊢ Eq (f (g ↑x)) ↑x
                            -/
        show f (g x) = x by rw [← hx', h x'] }
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem ofLeftInverse'_apply {g : B → A} {f : F} (h : Function.LeftInverse g f) (x : A) :
    ofLeftInverse' h x = f x :=
  rfl


@[simp]
theorem ofLeftInverse'_symm_apply {g : B → A} {f : F} (h : Function.LeftInverse g f)
    (x : NonUnitalStarAlgHom.range f) : (ofLeftInverse' h).symm x = g x :=
  rfl


/-- Restrict an injective non-unital star algebra homomorphism to a star algebra isomorphism -/
noncomputable def ofInjective' (f : F) (hf : Function.Injective f) :
    A ≃⋆ₐ[R] NonUnitalStarAlgHom.range f :=
  ofLeftInverse' (Classical.choose_spec hf.hasLeftInverse)


@[simp]
theorem ofInjective'_apply (f : F) (hf : Function.Injective f) (x : A) :
    ofInjective' f hf x = f x :=
  rfl


/-- The pointwise `star` of a non-unital subalgebra is a non-unital subalgebra. -/
instance instInvolutiveStar : InvolutiveStar (NonUnitalSubalgebra R A) where
  star S :=
    { carrier := star S.carrier
      mul_mem' := @fun x y hx hy => by simpa only [Set.mem_star, NonUnitalSubalgebra.mem_carrier]
        using (star_mul x y).symm ▸ mul_mem hy hx
      add_mem' := @fun x y hx hy => by simpa only [Set.mem_star, NonUnitalSubalgebra.mem_carrier]
        using (star_add x y).symm ▸ add_mem hx hy
      zero_mem' := Set.mem_star.mp ((star_zero A).symm ▸ zero_mem S : star (0 : A) ∈ S)
      smul_mem' := fun r x hx => by simpa only [Set.mem_star, NonUnitalSubalgebra.mem_carrier]
        using (star_smul r x).symm ▸ SMulMemClass.smul_mem (star r) hx }
  star_involutive S := NonUnitalSubalgebra.ext fun x =>
      ⟨fun hx => star_star x ▸ hx, fun hx => ((star_star x).symm ▸ hx : star (star x) ∈ S)⟩


@[simp]
theorem mem_star_iff (S : NonUnitalSubalgebra R A) (x : A) : x ∈ star S ↔ star x ∈ S :=
  Iff.rfl


theorem star_mem_star_iff (S : NonUnitalSubalgebra R A) (x : A) : star x ∈ star S ↔ x ∈ S := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : NonUnitalSemiring A
    inst✝² : StarRing A
    inst✝¹ : Module R A
    inst✝ : StarModule R A
    S : NonUnitalSubalgebra R A
    x : A
    ⊢ Iff (Membership.mem (Star.star S) (Star.star x)) (Membership.mem S x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_star (S : NonUnitalSubalgebra R A) : star S = star (S : Set A) :=
  rfl


theorem star_mono : Monotone (star : NonUnitalSubalgebra R A → NonUnitalSubalgebra R A) :=
  fun _ _ h _ hx => h hx


/-- The star operation on `NonUnitalSubalgebra` commutes with `NonUnitalAlgebra.adjoin`. -/
theorem star_adjoin_comm (s : Set A) :
    star (NonUnitalAlgebra.adjoin R s) = NonUnitalAlgebra.adjoin R (star s) :=
  have this :
    ∀ t : Set A, NonUnitalAlgebra.adjoin R (star t) ≤ star (NonUnitalAlgebra.adjoin R t) := fun _ =>
    NonUnitalAlgebra.adjoin_le fun _ hx => NonUnitalAlgebra.subset_adjoin R hx
                  /-
                    R : Type u
                    A : Type v
                    inst✝⁷ : CommSemiring R
                    inst✝⁶ : StarRing R
                    inst✝⁵ : NonUnitalSemiring A
                    inst✝⁴ : StarRing A
                    inst✝³ : Module R A
                    inst✝² : StarModule R A
                    inst✝¹ : IsScalarTower R A A
                    inst✝ : SMulCommClass R A A
                    s : Set A
                    this : ∀ (t : Set A), LE.le (NonUnitalAlgebra.adjoin R (Star.star t)) (Star.st …
                    ⊢ LE.le (Star.star (NonUnitalAlgebra.adjoin R s)) (NonUnitalAlgebra.adjoin R ( …
                  -/
  le_antisymm (by simpa only [star_star] using NonUnitalSubalgebra.star_mono (this (star s)))
                  /-
                    🎉 no goals
                  -/
    (this s)


/-- The `NonUnitalStarSubalgebra` obtained from `S : NonUnitalSubalgebra R A` by taking the
smallest non-unital subalgebra containing both `S` and `star S`. -/
@[simps!]
def starClosure (S : NonUnitalSubalgebra R A) : NonUnitalStarSubalgebra R A where
  toNonUnitalSubalgebra := S ⊔ star S
  star_mem' := @fun a (ha : a ∈ S ⊔ star S) => show star a ∈ S ⊔ star S by
    /-
      F : Type v'
      R' : Type u'
      R : Type u
      A : Type v
      B : Type w
      C : Type w'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : StarRing R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : StarRing A
      inst✝³ : Module R A
      inst✝² : StarModule R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      S : NonUnitalSubalgebra R A
      a : A
      ha : Membership.mem (Max.max S (Star.star S)) a
      ⊢ Membership.mem (Max.max S (Star.star S)) (Star.star a)
    -/
    simp only [← mem_star_iff _ a, ← (@NonUnitalAlgebra.gi R A _ _ _ _ _).l_sup_u _ _] at *
    /-
      F : Type v'
      R' : Type u'
      R : Type u
      A : Type v
      B : Type w
      C : Type w'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : StarRing R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : StarRing A
      inst✝³ : Module R A
      inst✝² : StarModule R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      S : NonUnitalSubalgebra R A
      a : A
      ha : Membership.mem (NonUnitalAlgebra.adjoin R (Max.max ↑S ↑(Star.star S))) a
      ⊢ Membership.mem (Star.star (NonUnitalAlgebra.adjoin R (Max.max ↑S ↑(Star.star …
    -/
    convert ha using 2
    simp only [Set.sup_eq_union, star_adjoin_comm, Set.union_star, coe_star, star_star,
      Set.union_comm]


theorem starClosure_le {S₁ : NonUnitalSubalgebra R A} {S₂ : NonUnitalStarSubalgebra R A}
    (h : S₁ ≤ S₂.toNonUnitalSubalgebra) : S₁.starClosure ≤ S₂ :=
  NonUnitalStarSubalgebra.toNonUnitalSubalgebra_le_iff.1 <|
    sup_le h fun x hx =>
      (star_star x ▸ star_mem (show star x ∈ S₂ from h <| (S₁.mem_star_iff _).1 hx) : x ∈ S₂)


theorem starClosure_le_iff {S₁ : NonUnitalSubalgebra R A} {S₂ : NonUnitalStarSubalgebra R A} :
    S₁.starClosure ≤ S₂ ↔ S₁ ≤ S₂.toNonUnitalSubalgebra :=
  ⟨fun h => le_sup_left.trans h, starClosure_le⟩


@[simp]
theorem starClosure_toNonunitalSubalgebra {S : NonUnitalSubalgebra R A} :
    S.starClosure.toNonUnitalSubalgebra = S ⊔ star S :=
  rfl


@[mono]
theorem starClosure_mono : Monotone (starClosure (R := R) (A := A)) :=
  fun _ _ h => starClosure_le <| h.trans le_sup_left


/-- The minimal non-unital subalgebra that includes `s`. -/
def adjoin (s : Set A) : NonUnitalStarSubalgebra R A where
  toNonUnitalSubalgebra := NonUnitalAlgebra.adjoin R (s ∪ star s)
  star_mem' _ := by
    rwa [NonUnitalSubalgebra.mem_carrier, ← NonUnitalSubalgebra.mem_star_iff,
      NonUnitalSubalgebra.star_adjoin_comm, Set.union_star, star_star, Set.union_comm]


theorem adjoin_eq_starClosure_adjoin (s : Set A) :
    adjoin R s = (NonUnitalAlgebra.adjoin R s).starClosure :=
  toNonUnitalSubalgebra_injective <| show
    NonUnitalAlgebra.adjoin R (s ∪ star s) =
      NonUnitalAlgebra.adjoin R s ⊔ star (NonUnitalAlgebra.adjoin R s)
    from
      (NonUnitalSubalgebra.star_adjoin_comm R s).symm ▸ NonUnitalAlgebra.adjoin_union s (star s)


theorem adjoin_toNonUnitalSubalgebra (s : Set A) :
    (adjoin R s).toNonUnitalSubalgebra = NonUnitalAlgebra.adjoin R (s ∪ star s) :=
  rfl


@[aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_adjoin (s : Set A) : s ⊆ adjoin R s :=
  Set.subset_union_left.trans <| NonUnitalAlgebra.subset_adjoin R


theorem star_subset_adjoin (s : Set A) : star s ⊆ adjoin R s :=
  Set.subset_union_right.trans <| NonUnitalAlgebra.subset_adjoin R


theorem self_mem_adjoin_singleton (x : A) : x ∈ adjoin R ({x} : Set A) :=
  NonUnitalAlgebra.subset_adjoin R <| Set.mem_union_left _ (Set.mem_singleton x)


theorem star_self_mem_adjoin_singleton (x : A) : star x ∈ adjoin R ({x} : Set A) :=
  star_mem <| self_mem_adjoin_singleton R x


@[elab_as_elim]
lemma adjoin_induction {s : Set A} {p : (x : A) → x ∈ adjoin R s → Prop}
    (mem : ∀ (x : A) (hx : x ∈ s), p x (subset_adjoin R s hx))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (zero : p 0 (zero_mem _)) (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    (smul : ∀ (r : R) x hx, p x hx → p (r • x) (SMulMemClass.smul_mem r hx))
    (star : ∀ x hx, p x hx → p (star x) (star_mem hx))
    {a : A} (ha : a ∈ adjoin R s) : p a ha := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (NonUnitalStarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
    add : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    zero : p 0 ⋯
    mul : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    smul : ∀ (r : R) (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s …
    star : ∀ (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x), p  …
    a : A
    ha : Membership.mem (NonUnitalStarAlgebra.adjoin R s) a
    ⊢ p a ha
  -/
  refine NonUnitalAlgebra.adjoin_induction (fun x hx ↦ ?_) add zero mul smul ha
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (NonUnitalStarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
    add : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    zero : p 0 ⋯
    mul : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    smul : ∀ (r : R) (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s …
    star : ∀ (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x), p  …
    a : A
    ha : Membership.mem (NonUnitalStarAlgebra.adjoin R s) a
    x : A
    hx : Membership.mem (Union.union s (Star.star s)) x
    ⊢ p x ⋯
  -/
  simp only [Set.mem_union, Set.mem_star] at hx
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (NonUnitalStarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
    add : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    zero : p 0 ⋯
    mul : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
    smul : ∀ (r : R) (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s …
    star : ∀ (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x), p  …
    a : A
    ha : Membership.mem (NonUnitalStarAlgebra.adjoin R s) a
    x : A
    hx✝ : Membership.mem (Union.union s (Star.star s)) x
    hx : Or (Membership.mem s x) (Membership.mem s (Star.star x))
    ⊢ p x ⋯
  -/
  obtain (hx | hx) := hx
    /-
      case inl
      R : Type u
      A : Type v
      inst✝⁷ : CommSemiring R
      inst✝⁶ : StarRing R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : StarRing A
      inst✝³ : Module R A
      inst✝² : IsScalarTower R A A
      inst✝¹ : SMulCommClass R A A
      inst✝ : StarModule R A
      s : Set A
      p : (x : A) → Membership.mem (NonUnitalStarAlgebra.adjoin R s) x → Prop
      mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
      add : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
      zero : p 0 ⋯
      mul : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
      smul : ∀ (r : R) (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s …
      star : ∀ (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x), p  …
      a : A
      ha : Membership.mem (NonUnitalStarAlgebra.adjoin R s) a
      x : A
      hx✝ : Membership.mem (Union.union s (Star.star s)) x
      hx : Membership.mem s x
      ⊢ p x ⋯
    -/
  · exact mem x hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      A : Type v
      inst✝⁷ : CommSemiring R
      inst✝⁶ : StarRing R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : StarRing A
      inst✝³ : Module R A
      inst✝² : IsScalarTower R A A
      inst✝¹ : SMulCommClass R A A
      inst✝ : StarModule R A
      s : Set A
      p : (x : A) → Membership.mem (NonUnitalStarAlgebra.adjoin R s) x → Prop
      mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
      add : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
      zero : p 0 ⋯
      mul : ∀ (x y : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x) (h …
      smul : ∀ (r : R) (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s …
      star : ∀ (x : A) (hx : Membership.mem (NonUnitalStarAlgebra.adjoin R s) x), p  …
      a : A
      ha : Membership.mem (NonUnitalStarAlgebra.adjoin R s) a
      x : A
      hx✝ : Membership.mem (Union.union s (Star.star s)) x
      hx : Membership.mem s (Star.star x)
      ⊢ p x ⋯
    -/
  · simpa using star _ (NonUnitalAlgebra.subset_adjoin R (by simpa using Or.inl hx)) (mem _ hx)
    /-
      🎉 no goals
    -/


@[deprecated adjoin_induction (since := "2024-10-10")]
alias adjoin_induction' := adjoin_induction


protected theorem gc : GaloisConnection (adjoin R : Set A → NonUnitalStarSubalgebra R A) (↑) := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    ⊢ GaloisConnection (NonUnitalStarAlgebra.adjoin R) SetLike.coe
  -/
  intro s S
  rw [← toNonUnitalSubalgebra_le_iff, adjoin_toNonUnitalSubalgebra,
    NonUnitalAlgebra.adjoin_le_iff, coe_toNonUnitalSubalgebra]
  exact ⟨fun h => Set.subset_union_left.trans h,
    fun h => Set.union_subset h fun x hx => star_star x ▸ star_mem (show star x ∈ S from h hx)⟩


/-- Galois insertion between `adjoin` and `Subtype.val`. -/
protected def gi : GaloisInsertion (adjoin R : Set A → NonUnitalStarSubalgebra R A) (↑) where
  choice s hs := (adjoin R s).copy s <| le_antisymm (NonUnitalStarAlgebra.gc.le_u_l s) hs
  gc := NonUnitalStarAlgebra.gc
  le_l_u S := (NonUnitalStarAlgebra.gc (S : Set A) (adjoin R S)).1 <| le_rfl
  choice_eq _ _ := NonUnitalStarSubalgebra.copy_eq _ _ _


theorem adjoin_le {S : NonUnitalStarSubalgebra R A} {s : Set A} (hs : s ⊆ S) : adjoin R s ≤ S :=
  NonUnitalStarAlgebra.gc.l_le hs


theorem adjoin_le_iff {S : NonUnitalStarSubalgebra R A} {s : Set A} : adjoin R s ≤ S ↔ s ⊆ S :=
  NonUnitalStarAlgebra.gc _ _


lemma adjoin_eq (s : NonUnitalStarSubalgebra R A) : adjoin R (s : Set A) = s :=
  le_antisymm (adjoin_le le_rfl) (subset_adjoin R (s : Set A))


lemma adjoin_eq_span (s : Set A) :
    (adjoin R s).toSubmodule = Submodule.span R (Subsemigroup.closure (s ∪ star s)) := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (NonUnitalStarAlgebra.adjoin R s).toSubmodule (Submodule.span R ↑(Subsemi …
  -/
  rw [adjoin_toNonUnitalSubalgebra, NonUnitalAlgebra.adjoin_eq_span]
  /-
    🎉 no goals
  -/


@[simp]
lemma span_eq_toSubmodule {R} [CommSemiring R] [Module R A] (s : NonUnitalStarSubalgebra R A) :
    Submodule.span R (s : Set A) = s.toSubmodule := by
  /-
    A : Type v
    inst✝³ : NonUnitalSemiring A
    inst✝² : StarRing A
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Module R A
    s : NonUnitalStarSubalgebra R A
    ⊢ Eq (Submodule.span R ↑s) s.toSubmodule
  -/
  simp [SetLike.ext'_iff, Submodule.coe_span_eq_self]
  /-
    🎉 no goals
  -/


theorem _root_.NonUnitalSubalgebra.starClosure_eq_adjoin (S : NonUnitalSubalgebra R A) :
    S.starClosure = adjoin R (S : Set A) :=
  le_antisymm (NonUnitalSubalgebra.starClosure_le_iff.2 <| subset_adjoin R (S : Set A))
    (adjoin_le (le_sup_left : S ≤ S ⊔ star S))


instance : CompleteLattice (NonUnitalStarSubalgebra R A) :=
  GaloisInsertion.liftCompleteLattice NonUnitalStarAlgebra.gi


@[simp]
theorem coe_top : ((⊤ : NonUnitalStarSubalgebra R A) : Set A) = Set.univ :=
  rfl


@[simp]
theorem mem_top {x : A} : x ∈ (⊤ : NonUnitalStarSubalgebra R A) :=
  Set.mem_univ x


@[simp]
theorem top_toNonUnitalSubalgebra :
                                                                      /-
                                                                        R : Type u
                                                                        A : Type v
                                                                        inst✝⁷ : CommSemiring R
                                                                        inst✝⁶ : StarRing R
                                                                        inst✝⁵ : NonUnitalSemiring A
                                                                        inst✝⁴ : StarRing A
                                                                        inst✝³ : Module R A
                                                                        inst✝² : IsScalarTower R A A
                                                                        inst✝¹ : SMulCommClass R A A
                                                                        inst✝ : StarModule R A
                                                                        ⊢ Eq Top.top.toNonUnitalSubalgebra Top.top
                                                                      -/
    (⊤ : NonUnitalStarSubalgebra R A).toNonUnitalSubalgebra = ⊤ := by ext; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem toNonUnitalSubalgebra_eq_top {S : NonUnitalStarSubalgebra R A} :
    S.toNonUnitalSubalgebra = ⊤ ↔ S = ⊤ :=
  NonUnitalStarSubalgebra.toNonUnitalSubalgebra_injective.eq_iff' top_toNonUnitalSubalgebra


theorem mem_sup_left {S T : NonUnitalStarSubalgebra R A} : ∀ {x : A}, x ∈ S → x ∈ S ⊔ T := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    S T : NonUnitalStarSubalgebra R A
    ⊢ ∀ {x : A}, Membership.mem S x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    S T : NonUnitalStarSubalgebra R A
    ⊢ LE.le S (Max.max S T)
  -/
  exact le_sup_left
  /-
    🎉 no goals
  -/


theorem mem_sup_right {S T : NonUnitalStarSubalgebra R A} : ∀ {x : A}, x ∈ T → x ∈ S ⊔ T := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    S T : NonUnitalStarSubalgebra R A
    ⊢ ∀ {x : A}, Membership.mem T x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    S T : NonUnitalStarSubalgebra R A
    ⊢ LE.le T (Max.max S T)
  -/
  exact le_sup_right
  /-
    🎉 no goals
  -/


theorem mul_mem_sup {S T : NonUnitalStarSubalgebra R A} {x y : A} (hx : x ∈ S) (hy : y ∈ T) :
    x * y ∈ S ⊔ T :=
  mul_mem (mem_sup_left hx) (mem_sup_right hy)


theorem map_sup [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B] (f : F)
    (S T : NonUnitalStarSubalgebra R A) :
    ((S ⊔ T).map f : NonUnitalStarSubalgebra R B) = S.map f ⊔ T.map f :=
  (NonUnitalStarSubalgebra.gc_map_comap f).l_sup


theorem map_inf [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B] (f : F)
    (hf : Function.Injective f) (S T : NonUnitalStarSubalgebra R A) :
    ((S ⊓ T).map f : NonUnitalStarSubalgebra R B) = S.map f ⊓ T.map f :=
  SetLike.coe_injective (Set.image_inter hf)


@[simp, norm_cast]
theorem coe_inf (S T : NonUnitalStarSubalgebra R A) : (↑(S ⊓ T) : Set A) = (S : Set A) ∩ T :=
  rfl


@[simp]
theorem mem_inf {S T : NonUnitalStarSubalgebra R A} {x : A} : x ∈ S ⊓ T ↔ x ∈ S ∧ x ∈ T :=
  Iff.rfl


@[simp]
theorem inf_toNonUnitalSubalgebra (S T : NonUnitalStarSubalgebra R A) :
    (S ⊓ T).toNonUnitalSubalgebra = S.toNonUnitalSubalgebra ⊓ T.toNonUnitalSubalgebra :=
  SetLike.coe_injective <| coe_inf _ _
  -- it's a bit surprising `rfl` fails here.


@[simp, norm_cast]
theorem coe_sInf (S : Set (NonUnitalStarSubalgebra R A)) : (↑(sInf S) : Set A) = ⋂ s ∈ S, ↑s :=
  sInf_image


theorem mem_sInf {S : Set (NonUnitalStarSubalgebra R A)} {x : A} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    S : Set (NonUnitalStarSubalgebra R A)
    x : A
    ⊢ Iff (Membership.mem (InfSet.sInf S) x) (∀ (p : NonUnitalStarSubalgebra R A), …
  -/
  simp only [← SetLike.mem_coe, coe_sInf, Set.mem_iInter₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem sInf_toNonUnitalSubalgebra (S : Set (NonUnitalStarSubalgebra R A)) :
    (sInf S).toNonUnitalSubalgebra = sInf (NonUnitalStarSubalgebra.toNonUnitalSubalgebra '' S) :=
                              /-
                                R : Type u
                                A : Type v
                                inst✝⁷ : CommSemiring R
                                inst✝⁶ : StarRing R
                                inst✝⁵ : NonUnitalSemiring A
                                inst✝⁴ : StarRing A
                                inst✝³ : Module R A
                                inst✝² : IsScalarTower R A A
                                inst✝¹ : SMulCommClass R A A
                                inst✝ : StarModule R A
                                S : Set (NonUnitalStarSubalgebra R A)
                                ⊢ Eq ↑(InfSet.sInf S).toNonUnitalSubalgebra ↑(InfSet.sInf (Set.image NonUnital …
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → NonUnitalStarSubalgebra R A} :
                                           /-
                                             R : Type u
                                             A : Type v
                                             inst✝⁷ : CommSemiring R
                                             inst✝⁶ : StarRing R
                                             inst✝⁵ : NonUnitalSemiring A
                                             inst✝⁴ : StarRing A
                                             inst✝³ : Module R A
                                             inst✝² : IsScalarTower R A A
                                             inst✝¹ : SMulCommClass R A A
                                             inst✝ : StarModule R A
                                             ι : Sort u_1
                                             S : ι → NonUnitalStarSubalgebra R A
                                             ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
                                           -/
    (↑(⨅ i, S i) : Set A) = ⋂ i, S i := by simp [iInf]
                                           /-
                                             🎉 no goals
                                           -/


theorem mem_iInf {ι : Sort*} {S : ι → NonUnitalStarSubalgebra R A} {x : A} :
                                        /-
                                          R : Type u
                                          A : Type v
                                          inst✝⁷ : CommSemiring R
                                          inst✝⁶ : StarRing R
                                          inst✝⁵ : NonUnitalSemiring A
                                          inst✝⁴ : StarRing A
                                          inst✝³ : Module R A
                                          inst✝² : IsScalarTower R A A
                                          inst✝¹ : SMulCommClass R A A
                                          inst✝ : StarModule R A
                                          ι : Sort u_1
                                          S : ι → NonUnitalStarSubalgebra R A
                                          x : A
                                          ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
                                        -/
    (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by simp only [iInf, mem_sInf, Set.forall_mem_range]
                                        /-
                                          🎉 no goals
                                        -/


theorem map_iInf {ι : Sort*} [Nonempty ι]
    [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B] (f : F)
    (hf : Function.Injective f) (S : ι → NonUnitalStarSubalgebra R A) :
    ((⨅ i, S i).map f : NonUnitalStarSubalgebra R B) = ⨅ i, (S i).map f := by
  /-
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : NonUnitalSemiring A
    inst✝¹⁴ : StarRing A
    inst✝¹³ : Module R A
    inst✝¹² : NonUnitalSemiring B
    inst✝¹¹ : StarRing B
    inst✝¹⁰ : Module R B
    inst✝⁹ : FunLike F A B
    inst✝⁸ : NonUnitalAlgHomClass F R A B
    inst✝⁷ : StarHomClass F A B
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : StarModule R A
    ι : Sort u_1
    inst✝³ : Nonempty ι
    inst✝² : IsScalarTower R B B
    inst✝¹ : SMulCommClass R B B
    inst✝ : StarModule R B
    f : F
    hf : Function.Injective ⇑f
    S : ι → NonUnitalStarSubalgebra R A
    ⊢ Eq (NonUnitalStarSubalgebra.map f (iInf fun i => S i)) (iInf fun i => NonUni …
  -/
  apply SetLike.coe_injective
  /-
    case a
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : NonUnitalSemiring A
    inst✝¹⁴ : StarRing A
    inst✝¹³ : Module R A
    inst✝¹² : NonUnitalSemiring B
    inst✝¹¹ : StarRing B
    inst✝¹⁰ : Module R B
    inst✝⁹ : FunLike F A B
    inst✝⁸ : NonUnitalAlgHomClass F R A B
    inst✝⁷ : StarHomClass F A B
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : StarModule R A
    ι : Sort u_1
    inst✝³ : Nonempty ι
    inst✝² : IsScalarTower R B B
    inst✝¹ : SMulCommClass R B B
    inst✝ : StarModule R B
    f : F
    hf : Function.Injective ⇑f
    S : ι → NonUnitalStarSubalgebra R A
    ⊢ Eq ↑(NonUnitalStarSubalgebra.map f (iInf fun i => S i)) ↑(iInf fun i => NonU …
  -/
  simpa using (Set.injOn_of_injective hf).image_iInter_eq (s := SetLike.coe ∘ S)
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_toNonUnitalSubalgebra {ι : Sort*} (S : ι → NonUnitalStarSubalgebra R A) :
    (⨅ i, S i).toNonUnitalSubalgebra = ⨅ i, (S i).toNonUnitalSubalgebra :=
                              /-
                                R : Type u
                                A : Type v
                                inst✝⁷ : CommSemiring R
                                inst✝⁶ : StarRing R
                                inst✝⁵ : NonUnitalSemiring A
                                inst✝⁴ : StarRing A
                                inst✝³ : Module R A
                                inst✝² : IsScalarTower R A A
                                inst✝¹ : SMulCommClass R A A
                                inst✝ : StarModule R A
                                ι : Sort u_1
                                S : ι → NonUnitalStarSubalgebra R A
                                ⊢ Eq ↑(iInf fun i => S i).toNonUnitalSubalgebra ↑(iInf fun i => (S i).toNonUni …
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


instance : Inhabited (NonUnitalStarSubalgebra R A) :=
  ⟨⊥⟩


theorem mem_bot {x : A} : x ∈ (⊥ : NonUnitalStarSubalgebra R A) ↔ x = 0 :=
  show x ∈ NonUnitalAlgebra.adjoin R (∅ ∪ star ∅ : Set A) ↔ x = 0 by
    /-
      R : Type u
      A : Type v
      inst✝⁷ : CommSemiring R
      inst✝⁶ : StarRing R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : StarRing A
      inst✝³ : Module R A
      inst✝² : IsScalarTower R A A
      inst✝¹ : SMulCommClass R A A
      inst✝ : StarModule R A
      x : A
      ⊢ Iff (Membership.mem (NonUnitalAlgebra.adjoin R (Union.union EmptyCollection. …
    -/
    rw [Set.star_empty, Set.union_empty, NonUnitalAlgebra.adjoin_empty, NonUnitalAlgebra.mem_bot]
    /-
      🎉 no goals
    -/


theorem toNonUnitalSubalgebra_bot :
    (⊥ : NonUnitalStarSubalgebra R A).toNonUnitalSubalgebra = ⊥ := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    ⊢ Eq Bot.bot.toNonUnitalSubalgebra Bot.bot
  -/
  ext x
  /-
    case h
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    x : A
    ⊢ Iff (Membership.mem Bot.bot.toNonUnitalSubalgebra x) (Membership.mem Bot.bot …
  -/
  simp only [mem_bot, NonUnitalStarSubalgebra.mem_toNonUnitalSubalgebra, NonUnitalAlgebra.mem_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_bot : ((⊥ : NonUnitalStarSubalgebra R A) : Set A) = {0} := by
  simp only [Set.ext_iff, NonUnitalStarAlgebra.mem_bot, SetLike.mem_coe, Set.mem_singleton_iff,
    forall_const]


theorem eq_top_iff {S : NonUnitalStarSubalgebra R A} : S = ⊤ ↔ ∀ x : A, x ∈ S :=
                 /-
                   R : Type u
                   A : Type v
                   inst✝⁷ : CommSemiring R
                   inst✝⁶ : StarRing R
                   inst✝⁵ : NonUnitalSemiring A
                   inst✝⁴ : StarRing A
                   inst✝³ : Module R A
                   inst✝² : IsScalarTower R A A
                   inst✝¹ : SMulCommClass R A A
                   inst✝ : StarModule R A
                   S : NonUnitalStarSubalgebra R A
                   h : Eq S Top.top
                   x : A
                   ⊢ Membership.mem S x
                 -/
  ⟨fun h x => by rw [h]; exact mem_top,
                         /-
                           🎉 no goals
                         -/
                /-
                  R : Type u
                  A : Type v
                  inst✝⁷ : CommSemiring R
                  inst✝⁶ : StarRing R
                  inst✝⁵ : NonUnitalSemiring A
                  inst✝⁴ : StarRing A
                  inst✝³ : Module R A
                  inst✝² : IsScalarTower R A A
                  inst✝¹ : SMulCommClass R A A
                  inst✝ : StarModule R A
                  S : NonUnitalStarSubalgebra R A
                  h : ∀ (x : A), Membership.mem S x
                  ⊢ Eq S Top.top
                -/
    fun h => by ext x; exact ⟨fun _ => mem_top, fun _ => h x⟩⟩
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem range_id : NonUnitalStarAlgHom.range (NonUnitalStarAlgHom.id R A) = ⊤ :=
  SetLike.coe_injective Set.range_id


@[simp]
theorem map_bot [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B] (f : F) :
    (⊥ : NonUnitalStarSubalgebra R A).map f = ⊥ :=
                              /-
                                F : Type v'
                                R : Type u
                                A : Type v
                                B : Type w
                                inst✝¹⁶ : CommSemiring R
                                inst✝¹⁵ : StarRing R
                                inst✝¹⁴ : NonUnitalSemiring A
                                inst✝¹³ : StarRing A
                                inst✝¹² : Module R A
                                inst✝¹¹ : NonUnitalSemiring B
                                inst✝¹⁰ : StarRing B
                                inst✝⁹ : Module R B
                                inst✝⁸ : FunLike F A B
                                inst✝⁷ : NonUnitalAlgHomClass F R A B
                                inst✝⁶ : StarHomClass F A B
                                inst✝⁵ : IsScalarTower R A A
                                inst✝⁴ : SMulCommClass R A A
                                inst✝³ : StarModule R A
                                inst✝² : IsScalarTower R B B
                                inst✝¹ : SMulCommClass R B B
                                inst✝ : StarModule R B
                                f : F
                                ⊢ Eq ↑(NonUnitalStarSubalgebra.map f Bot.bot) ↑Bot.bot
                              -/
  SetLike.coe_injective <| by simp [NonUnitalAlgebra.coe_bot, NonUnitalStarSubalgebra.coe_map]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem comap_top [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B] (f : F) :
    (⊤ : NonUnitalStarSubalgebra R B).comap f = ⊤ :=
  eq_top_iff.2 fun _x => mem_top


/-- `NonUnitalStarAlgHom` to `⊤ : NonUnitalStarSubalgebra R A`. -/
def toTop : A →⋆ₙₐ[R] (⊤ : NonUnitalStarSubalgebra R A) :=
  NonUnitalStarAlgHom.codRestrict (NonUnitalStarAlgHom.id R A) ⊤ fun _ => mem_top


theorem range_eq_top [IsScalarTower R B B] [SMulCommClass R B B] [StarModule R B]
    (f : F) : NonUnitalStarAlgHom.range f = (⊤ : NonUnitalStarSubalgebra R B) ↔
      Function.Surjective f :=
  NonUnitalStarAlgebra.eq_top_iff


@[deprecated (since := "2024-11-11")] alias range_top_iff_surjective := range_eq_top


@[simp]
theorem map_top [IsScalarTower R A A] [SMulCommClass R A A] [StarModule R A] (f : F) :
    (⊤ : NonUnitalStarSubalgebra R A).map f = NonUnitalStarAlgHom.range f :=
  SetLike.coe_injective Set.image_univ


lemma _root_.NonUnitalStarAlgHom.map_adjoin (f : F) (s : Set A) :
    map f (adjoin R s) = adjoin R (f '' s) :=
  Set.image_preimage.l_comm_of_u_comm (gc_map_comap f) NonUnitalStarAlgebra.gi.gc
    NonUnitalStarAlgebra.gi.gc fun _t => rfl


@[simp]
lemma _root_.NonUnitalStarAlgHom.map_adjoin_singleton (f : F) (x : A) :
    map f (adjoin R {x}) = adjoin R {f x} := by
  /-
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : NonUnitalSemiring A
    inst✝¹⁴ : StarRing A
    inst✝¹³ : Module R A
    inst✝¹² : NonUnitalSemiring B
    inst✝¹¹ : StarRing B
    inst✝¹⁰ : Module R B
    inst✝⁹ : FunLike F A B
    inst✝⁸ : NonUnitalAlgHomClass F R A B
    inst✝⁷ : StarHomClass F A B
    inst✝⁶ : StarRing R
    inst✝⁵ : IsScalarTower R A A
    inst✝⁴ : SMulCommClass R A A
    inst✝³ : StarModule R A
    inst✝² : IsScalarTower R B B
    inst✝¹ : SMulCommClass R B B
    inst✝ : StarModule R B
    f : F
    x : A
    ⊢ Eq (NonUnitalStarSubalgebra.map f (NonUnitalStarAlgebra.adjoin R (Singleton. …
  -/
  simp [NonUnitalStarAlgHom.map_adjoin]
  /-
    🎉 no goals
  -/


instance subsingleton_of_subsingleton [Subsingleton A] :
    Subsingleton (NonUnitalStarSubalgebra R A) :=
                              /-
                                F : Type v'
                                R' : Type u'
                                R : Type u
                                A : Type v
                                B✝ : Type w
                                C✝ : Type w'
                                inst✝¹⁷ : CommSemiring R
                                inst✝¹⁶ : NonUnitalSemiring A
                                inst✝¹⁵ : StarRing A
                                inst✝¹⁴ : Module R A
                                inst✝¹³ : NonUnitalSemiring B✝
                                inst✝¹² : StarRing B✝
                                inst✝¹¹ : Module R B✝
                                inst✝¹⁰ : FunLike F A B✝
                                inst✝⁹ : NonUnitalAlgHomClass F R A B✝
                                inst✝⁸ : StarHomClass F A B✝
                                S : NonUnitalStarSubalgebra R A
                                inst✝⁷ : StarRing R
                                inst✝⁶ : IsScalarTower R A A
                                inst✝⁵ : SMulCommClass R A A
                                inst✝⁴ : StarModule R A
                                inst✝³ : IsScalarTower R B✝ B✝
                                inst✝² : SMulCommClass R B✝ B✝
                                inst✝¹ : StarModule R B✝
                                inst✝ : Subsingleton A
                                B C : NonUnitalStarSubalgebra R A
                                x : A
                                ⊢ Iff (Membership.mem B x) (Membership.mem C x)
                              -/
  ⟨fun B C => ext fun x => by simp only [Subsingleton.elim x 0, zero_mem B, zero_mem C]⟩
                              /-
                                🎉 no goals
                              -/


instance _root_.NonUnitalStarAlgHom.subsingleton [Subsingleton (NonUnitalStarSubalgebra R A)] :
    Subsingleton (A →⋆ₙₐ[R] B) :=
  ⟨fun f g => NonUnitalStarAlgHom.ext fun a =>
    have : a ∈ (⊥ : NonUnitalStarSubalgebra R A) :=
      Subsingleton.elim (⊤ : NonUnitalStarSubalgebra R A) ⊥ ▸ mem_top
    (mem_bot.mp this).symm ▸ (map_zero f).trans (map_zero g).symm⟩


/--
The map `S → T` when `S` is a non-unital star subalgebra contained in the non-unital star
algebra `T`.

This is the non-unital star subalgebra version of `Submodule.inclusion`, or
`NonUnitalSubalgebra.inclusion`  -/
def inclusion {S T : NonUnitalStarSubalgebra R A} (h : S ≤ T) : S →⋆ₙₐ[R] T where
  toNonUnitalAlgHom := NonUnitalSubalgebra.inclusion h
  map_star' _ := rfl


theorem inclusion_injective {S T : NonUnitalStarSubalgebra R A} (h : S ≤ T) :
    Function.Injective (inclusion h) :=
  fun _ _ => Subtype.ext ∘ Subtype.mk.inj


@[simp]
theorem inclusion_self {S : NonUnitalStarSubalgebra R A} :
    inclusion (le_refl S) = NonUnitalAlgHom.id R S :=
  NonUnitalAlgHom.ext fun _x => Subtype.ext rfl


@[simp]
theorem inclusion_mk {S T : NonUnitalStarSubalgebra R A} (h : S ≤ T) (x : A) (hx : x ∈ S) :
    inclusion h ⟨x, hx⟩ = ⟨x, h hx⟩ :=
  rfl


theorem inclusion_right {S T : NonUnitalStarSubalgebra R A} (h : S ≤ T) (x : T) (m : (x : A) ∈ S) :
    inclusion h ⟨x, m⟩ = x :=
  Subtype.ext rfl


@[simp]
theorem inclusion_inclusion {S T U : NonUnitalStarSubalgebra R A} (hst : S ≤ T) (htu : T ≤ U)
    (x : S) : inclusion htu (inclusion hst x) = inclusion (le_trans hst htu) x :=
  Subtype.ext rfl


@[simp]
theorem val_inclusion {S T : NonUnitalStarSubalgebra R A} (h : S ≤ T) (s : S) :
    (inclusion h s : A) = s :=
  rfl


theorem range_val : NonUnitalStarAlgHom.range (NonUnitalStarSubalgebraClass.subtype S) = S :=
  ext <| Set.ext_iff.1 <| (NonUnitalStarSubalgebraClass.subtype S).coe_range.trans Subtype.range_val


/-- The product of two non-unital star subalgebras is a non-unital star subalgebra. -/
def prod : NonUnitalStarSubalgebra R (A × B) :=
  { S.toNonUnitalSubalgebra.prod S₁.toNonUnitalSubalgebra with
    carrier := S ×ˢ S₁
    star_mem' := fun hx => ⟨star_mem hx.1, star_mem hx.2⟩ }


@[simp]
theorem coe_prod : (prod S S₁ : Set (A × B)) = (S : Set A) ×ˢ S₁ :=
  rfl


theorem prod_toNonUnitalSubalgebra :
    (S.prod S₁).toNonUnitalSubalgebra = S.toNonUnitalSubalgebra.prod S₁.toNonUnitalSubalgebra :=
  rfl


@[simp]
theorem mem_prod {S : NonUnitalStarSubalgebra R A} {S₁ : NonUnitalStarSubalgebra R B} {x : A × B} :
    x ∈ prod S S₁ ↔ x.1 ∈ S ∧ x.2 ∈ S₁ :=
  Set.mem_prod


@[simp]
                                                                            /-
                                                                              R : Type u
                                                                              A : Type v
                                                                              B : Type w
                                                                              inst✝¹³ : CommSemiring R
                                                                              inst✝¹² : NonUnitalSemiring A
                                                                              inst✝¹¹ : StarRing A
                                                                              inst✝¹⁰ : Module R A
                                                                              inst✝⁹ : NonUnitalSemiring B
                                                                              inst✝⁸ : StarRing B
                                                                              inst✝⁷ : Module R B
                                                                              inst✝⁶ : StarRing R
                                                                              inst✝⁵ : IsScalarTower R A A
                                                                              inst✝⁴ : SMulCommClass R A A
                                                                              inst✝³ : StarModule R A
                                                                              inst✝² : IsScalarTower R B B
                                                                              inst✝¹ : SMulCommClass R B B
                                                                              inst✝ : StarModule R B
                                                                              ⊢ Eq (Top.top.prod Top.top) Top.top
                                                                            -/
theorem prod_top : (prod ⊤ ⊤ : NonUnitalStarSubalgebra R (A × B)) = ⊤ := by ext; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem prod_mono {S T : NonUnitalStarSubalgebra R A} {S₁ T₁ : NonUnitalStarSubalgebra R B} :
    S ≤ T → S₁ ≤ T₁ → prod S S₁ ≤ prod T T₁ :=
  Set.prod_mono


@[simp]
theorem prod_inf_prod {S T : NonUnitalStarSubalgebra R A} {S₁ T₁ : NonUnitalStarSubalgebra R B} :
    S.prod S₁ ⊓ T.prod T₁ = (S ⊓ T).prod (S₁ ⊓ T₁) :=
  SetLike.coe_injective Set.prod_inter_prod


theorem coe_iSup_of_directed [Nonempty ι] {S : ι → NonUnitalStarSubalgebra R A}
    (dir : Directed (· ≤ ·) S) : ↑(iSup S) = ⋃ i, (S i : Set A) :=
  let K : NonUnitalStarSubalgebra R A :=
    { __ := NonUnitalSubalgebra.copy _ _ (NonUnitalSubalgebra.coe_iSup_of_directed dir).symm
      star_mem' := fun hx ↦
        let ⟨i, hi⟩ := Set.mem_iUnion.1 hx
        Set.mem_iUnion.2 ⟨i, star_mem (s := S i) hi⟩ }
  have : iSup S = K := le_antisymm (iSup_le fun i ↦ le_iSup (fun i ↦ (S i : Set A)) i)
    (Set.iUnion_subset fun _ ↦ le_iSup S _)
  this.symm ▸ rfl


/-- Define a non-unital star algebra homomorphism on a directed supremum of non-unital star
subalgebras by defining it on each non-unital star subalgebra, and proving that it agrees on the
intersection of non-unital star subalgebras. -/
noncomputable def iSupLift [Nonempty ι] (K : ι → NonUnitalStarSubalgebra R A)
    (dir : Directed (· ≤ ·) K) (f : ∀ i, K i →⋆ₙₐ[R] B)
    (hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h))
    (T : NonUnitalStarSubalgebra R A) (hT : T = iSup K) : ↥T →⋆ₙₐ[R] B := by
  /-
    F : Type v'
    R' : Type u'
    R : Type u
    A : Type v
    B : Type w
    C : Type w'
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : NonUnitalSemiring A
    inst✝¹⁵ : StarRing A
    inst✝¹⁴ : Module R A
    inst✝¹³ : NonUnitalSemiring B
    inst✝¹² : StarRing B
    inst✝¹¹ : Module R B
    inst✝¹⁰ : FunLike F A B
    inst✝⁹ : NonUnitalAlgHomClass F R A B
    inst✝⁸ : StarHomClass F A B
    S : NonUnitalStarSubalgebra R A
    ι : Type u_1
    inst✝⁷ : StarRing R
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : StarModule R A
    inst✝³ : IsScalarTower R B B
    inst✝² : SMulCommClass R B B
    inst✝¹ : StarModule R B
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    T : NonUnitalStarSubalgebra R A
    hT : Eq T (iSup K)
    ⊢ NonUnitalStarAlgHom R (Subtype fun x => Membership.mem T x) B
  -/
  subst hT
  exact
    { toFun :=
        Set.iUnionLift (fun i => ↑(K i)) (fun i x => f i x)
          (fun i j x hxi hxj => by
            let ⟨k, hik, hjk⟩ := dir i j
            simp only
            rw [hf i k hik, hf j k hjk]
            rfl)
          _ (by rw [coe_iSup_of_directed dir])
      map_zero' := by
        dsimp only [SetLike.coe_sort_coe, NonUnitalAlgHom.coe_comp, Function.comp_apply,
          inclusion_mk, Eq.ndrec, id_eq, eq_mpr_eq_cast]
        exact Set.iUnionLift_const _ (fun i : ι => (0 : K i)) (fun _ => rfl)  _ (by simp)
      map_mul' := by
        dsimp only [SetLike.coe_sort_coe, NonUnitalAlgHom.coe_comp, Function.comp_apply,
          inclusion_mk, Eq.ndrec, id_eq, eq_mpr_eq_cast, ZeroMemClass.coe_zero,
          AddSubmonoid.mk_add_mk, Set.inclusion_mk]
        apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· * ·))
        all_goals simp
      map_add' := by
        dsimp only [SetLike.coe_sort_coe, NonUnitalAlgHom.coe_comp, Function.comp_apply,
          inclusion_mk, Eq.ndrec, id_eq, eq_mpr_eq_cast]
        apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· + ·))
        all_goals simp
      map_smul' := fun r => by
        dsimp only [SetLike.coe_sort_coe, NonUnitalAlgHom.coe_comp, Function.comp_apply,
          inclusion_mk, Eq.ndrec, id_eq, eq_mpr_eq_cast]
        apply Set.iUnionLift_unary (coe_iSup_of_directed dir) _ (fun _ x => r • x)
          (fun _ _ => rfl)
        all_goals simp
      map_star' := by
        dsimp only [SetLike.coe_sort_coe, NonUnitalStarAlgHom.comp_apply, inclusion_mk, Eq.ndrec,
          id_eq, eq_mpr_eq_cast, ZeroMemClass.coe_zero, AddSubmonoid.mk_add_mk, Set.inclusion_mk,
          MulMemClass.mk_mul_mk, NonUnitalAlgHom.toDistribMulActionHom_eq_coe,
          DistribMulActionHom.toFun_eq_coe, NonUnitalAlgHom.coe_to_distribMulActionHom,
          NonUnitalAlgHom.coe_mk]
        apply Set.iUnionLift_unary (coe_iSup_of_directed dir) _ (fun _ x => star x)
          (fun _ _ => rfl)
        all_goals simp [map_star] }


@[simp]
theorem iSupLift_inclusion {i : ι} (x : K i) (h : K i ≤ T) :
    iSupLift K dir f hf T hT (inclusion h x) = f i x := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    T : NonUnitalStarSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) T
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf T hT) ((NonUnitalStarSubalg …
  -/
  subst T
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) (iSup K)
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf (iSup K) ⋯) ((NonUnitalStar …
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) (iSup K)
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑(iSup K) ⋯ ((No …
  -/
  apply Set.iUnionLift_inclusion
  /-
    case h
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) (iSup K)
    ⊢ HasSubset.Subset ↑(K i) ↑(iSup K)
  -/
  exact h
  /-
    🎉 no goals
  -/


@[simp]
theorem iSupLift_comp_inclusion {i : ι} (h : K i ≤ T) :
                                                              /-
                                                                R : Type u
                                                                A : Type v
                                                                B : Type w
                                                                inst✝¹¹ : CommSemiring R
                                                                inst✝¹⁰ : NonUnitalSemiring A
                                                                inst✝⁹ : StarRing A
                                                                inst✝⁸ : Module R A
                                                                inst✝⁷ : NonUnitalSemiring B
                                                                inst✝⁶ : StarRing B
                                                                inst✝⁵ : Module R B
                                                                ι : Type u_1
                                                                inst✝⁴ : StarRing R
                                                                inst✝³ : IsScalarTower R A A
                                                                inst✝² : SMulCommClass R A A
                                                                inst✝¹ : StarModule R A
                                                                inst✝ : Nonempty ι
                                                                K : ι → NonUnitalStarSubalgebra R A
                                                                dir : Directed (fun x1 x2 => LE.le x1 x2) K
                                                                f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
                                                                hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
                                                                T : NonUnitalStarSubalgebra R A
                                                                hT : Eq T (iSup K)
                                                                i : ι
                                                                h : LE.le (K i) T
                                                                ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf T hT).comp (NonUnitalStarSu …
                                                              -/
    (iSupLift K dir f hf T hT).comp (inclusion h) = f i := by ext; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem iSupLift_mk {i : ι} (x : K i) (hx : (x : A) ∈ T) :
    iSupLift K dir f hf T hT ⟨x, hx⟩ = f i x := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    T : NonUnitalStarSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem T ↑x
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf T hT) ⟨↑x, hx⟩) ((f i) x)
  -/
  subst hT
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem (iSup K) ↑x
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf (iSup K) ⋯) ⟨↑x, hx⟩) ((f i …
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem (iSup K) ↑x
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑(iSup K) ⋯ ⟨↑x, …
  -/
  apply Set.iUnionLift_mk
  /-
    🎉 no goals
  -/


theorem iSupLift_of_mem {i : ι} (x : T) (hx : (x : A) ∈ K i) :
    iSupLift K dir f hf T hT x = f i ⟨x, hx⟩ := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    T : NonUnitalStarSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem T x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf T hT) x) ((f i) ⟨↑x, hx⟩)
  -/
  subst hT
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (iSup K) x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq ((NonUnitalStarSubalgebra.iSupLift K dir f hf (iSup K) ⋯) x) ((f i) ⟨↑x,  …
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalSemiring B
    inst✝⁶ : StarRing B
    inst✝⁵ : Module R B
    ι : Type u_1
    inst✝⁴ : StarRing R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : StarModule R A
    inst✝ : Nonempty ι
    K : ι → NonUnitalStarSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalStarAlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalStarS …
    i : ι
    x : Subtype fun x => Membership.mem (iSup K) x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑(iSup K) ⋯ x) ( …
  -/
  apply Set.iUnionLift_of_mem
  /-
    🎉 no goals
  -/


/-- The center of a non-unital star algebra is the set of elements which commute with every element.
They form a non-unital star subalgebra. -/
def center : NonUnitalStarSubalgebra R A where
  toNonUnitalSubalgebra := NonUnitalSubalgebra.center R A
  star_mem' := Set.star_mem_center


theorem coe_center : (center R A : Set A) = Set.center A :=
  rfl


@[simp]
theorem center_toNonUnitalSubalgebra :
    (center R A).toNonUnitalSubalgebra = NonUnitalSubalgebra.center R A :=
  rfl


@[simp]
theorem center_eq_top (A : Type*) [StarRing R] [NonUnitalCommSemiring A] [StarRing A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A] [StarModule R A] : center R A = ⊤ :=
  SetLike.coe_injective (Set.center_eq_univ A)


instance instNonUnitalCommSemiring : NonUnitalCommSemiring (center R A) :=
  NonUnitalSubalgebra.center.instNonUnitalCommSemiring


instance instNonUnitalCommRing {A : Type*} [NonUnitalRing A] [StarRing A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A] : NonUnitalCommRing (center R A) :=
  NonUnitalSubalgebra.center.instNonUnitalCommRing


theorem mem_center_iff {a : A} : a ∈ center R A ↔ ∀ b : A, b * a = a * b :=
  Subsemigroup.mem_center_iff


/-- The centralizer of the star-closure of a set as a non-unital star subalgebra. -/
def centralizer (s : Set A) : NonUnitalStarSubalgebra R A :=
  { NonUnitalSubalgebra.centralizer R (s ∪ star s) with
    star_mem' := Set.star_mem_centralizer }


@[simp, norm_cast]
theorem coe_centralizer (s : Set A) : (centralizer R s : Set A) = (s ∪ star s).centralizer :=
  rfl


theorem mem_centralizer_iff {s : Set A} {z : A} :
    z ∈ centralizer R s ↔ ∀ g ∈ s, g * z = z * g ∧ star g * z = z * star g := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : NonUnitalSemiring A
    inst✝³ : StarRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    s : Set A
    z : A
    ⊢ Iff (Membership.mem (NonUnitalStarSubalgebra.centralizer R s) z) (∀ (g : A), …
  -/
  show (∀ g ∈ s ∪ star s, g * z = z * g) ↔ ∀ g ∈ s, g * z = z * g ∧ star g * z = z * star g
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : NonUnitalSemiring A
    inst✝³ : StarRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    s : Set A
    z : A
    ⊢ Iff (∀ (g : A), Membership.mem (Union.union s (Star.star s)) g → Eq (HMul.hM …
  -/
  simp only [Set.mem_union, or_imp, forall_and, and_congr_right_iff]
  exact fun _ =>
    ⟨fun hz a ha => hz _ (Set.star_mem_star.mpr ha), fun hz a ha => star_star a ▸ hz _ ha⟩


theorem centralizer_le (s t : Set A) (h : s ⊆ t) : centralizer R t ≤ centralizer R s :=
  Set.centralizer_subset (Set.union_subset_union h <| Set.preimage_mono h)


@[simp]
theorem centralizer_univ : centralizer R Set.univ = center R A :=
                     /-
                       R : Type u
                       A : Type v
                       inst✝⁵ : CommSemiring R
                       inst✝⁴ : NonUnitalSemiring A
                       inst✝³ : StarRing A
                       inst✝² : Module R A
                       inst✝¹ : IsScalarTower R A A
                       inst✝ : SMulCommClass R A A
                       ⊢ Eq ↑(NonUnitalStarSubalgebra.centralizer R Set.univ) ↑(NonUnitalStarSubalgeb …
                     -/
  SetLike.ext' <| by rw [coe_centralizer, Set.univ_union, coe_center, Set.centralizer_univ]
                     /-
                       🎉 no goals
                     -/


theorem centralizer_toNonUnitalSubalgebra (s : Set A) :
    (centralizer R s).toNonUnitalSubalgebra = NonUnitalSubalgebra.centralizer R (s ∪ star s) :=
  rfl


theorem coe_centralizer_centralizer (s : Set A) :
    (centralizer R (centralizer R s : Set A)) = (s ∪ star s).centralizer.centralizer := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : NonUnitalSemiring A
    inst✝³ : StarRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    s : Set A
    ⊢ Eq (↑(NonUnitalStarSubalgebra.centralizer R ↑(NonUnitalStarSubalgebra.centra …
  -/
  rw [coe_centralizer, StarMemClass.star_coe_eq, Set.union_self, coe_centralizer]
  /-
    🎉 no goals
  -/


variable (R) in
lemma adjoin_le_centralizer_centralizer (s : Set A) :
    adjoin R s ≤ centralizer R (centralizer R s) := by
  rw [← toNonUnitalSubalgebra_le_iff, centralizer_toNonUnitalSubalgebra,
    adjoin_toNonUnitalSubalgebra]
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    ⊢ LE.le (NonUnitalAlgebra.adjoin R (Union.union s (Star.star s))) (NonUnitalSu …
  -/
  convert NonUnitalAlgebra.adjoin_le_centralizer_centralizer R (s ∪ star s)
  /-
    case h.e'_4.h.e'_8
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (Union.union (↑(NonUnitalStarSubalgebra.centralizer R s)) (Star.star ↑(No …
  -/
  rw [StarMemClass.star_coe_eq]
  /-
    case h.e'_4.h.e'_8
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : StarRing A
    inst✝³ : Module R A
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (Union.union ↑(NonUnitalStarSubalgebra.centralizer R s) ↑(NonUnitalStarSu …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma commute_of_mem_adjoin_of_forall_mem_commute {a b : A} {s : Set A}
    (hb : b ∈ adjoin R s) (h : ∀ b ∈ s, Commute a b) (h_star : ∀ b ∈ s, Commute a (star b)) :
    Commute a b :=
  NonUnitalAlgebra.commute_of_mem_adjoin_of_forall_mem_commute hb fun b hb ↦
                      /-
                        R : Type u
                        A : Type v
                        inst✝⁷ : CommSemiring R
                        inst✝⁶ : StarRing R
                        inst✝⁵ : NonUnitalSemiring A
                        inst✝⁴ : StarRing A
                        inst✝³ : Module R A
                        inst✝² : IsScalarTower R A A
                        inst✝¹ : SMulCommClass R A A
                        inst✝ : StarModule R A
                        a b✝ : A
                        s : Set A
                        hb✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) b✝
                        h : ∀ (b : A), Membership.mem s b → Commute a b
                        h_star : ∀ (b : A), Membership.mem s b → Commute a (Star.star b)
                        b : A
                        hb : Membership.mem (Union.union s (Star.star s)) b
                        ⊢ Membership.mem (Star.star s) b → Commute a b
                      -/
    hb.elim (h b) (by simpa using h_star (star b))
                      /-
                        🎉 no goals
                      -/


lemma commute_of_mem_adjoin_singleton_of_commute {a b c : A}
    (hc : c ∈ adjoin R {b}) (h : Commute a b) (h_star : Commute a (star b)) :
    Commute a c :=
                                                     /-
                                                       R : Type u
                                                       A : Type v
                                                       inst✝⁷ : CommSemiring R
                                                       inst✝⁶ : StarRing R
                                                       inst✝⁵ : NonUnitalSemiring A
                                                       inst✝⁴ : StarRing A
                                                       inst✝³ : Module R A
                                                       inst✝² : IsScalarTower R A A
                                                       inst✝¹ : SMulCommClass R A A
                                                       inst✝ : StarModule R A
                                                       a b c : A
                                                       hc : Membership.mem (NonUnitalStarAlgebra.adjoin R (Singleton.singleton b)) c
                                                       h : Commute a b
                                                       h_star : Commute a (Star.star b)
                                                       ⊢ ∀ (b_1 : A), Membership.mem (Singleton.singleton b) b_1 → Commute a b_1
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  commute_of_mem_adjoin_of_forall_mem_commute hc (by simpa) (by simpa)
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma commute_of_mem_adjoin_self {a b : A} [IsStarNormal a] (hb : b ∈ adjoin R {a}) :
    Commute a b :=
  commute_of_mem_adjoin_singleton_of_commute hb rfl (isStarNormal_iff a |>.mp inferInstance).symm


variable (R) in
/-- If all elements of `s : Set A` commute pairwise and with elements of `star s`, then `adjoin R s`
is a non-unital commutative semiring.

See note [reducible non-instances]. -/
abbrev adjoinNonUnitalCommSemiringOfComm {s : Set A} (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a)
    (hcomm_star : ∀ a ∈ s, ∀ b ∈ s, a * star b = star b * a) :
    NonUnitalCommSemiring (adjoin R s) :=
  { (adjoin R s).toNonUnitalSemiring with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦ by
      have hcomm : ∀ a ∈ s ∪ star s, ∀ b ∈ s ∪ star s, a * b = b * a := fun a ha b hb ↦
        Set.union_star_self_comm (fun _ ha _ hb ↦ hcomm _ hb _ ha)
          (fun _ ha _ hb ↦ hcomm_star _ hb _ ha) b hb a ha
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : StarRing R
        inst✝⁵ : NonUnitalSemiring A
        inst✝⁴ : StarRing A
        inst✝³ : Module R A
        inst✝² : IsScalarTower R A A
        inst✝¹ : SMulCommClass R A A
        inst✝ : StarModule R A
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalStarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁⟩)
      -/
      have := adjoin_le_centralizer_centralizer R s
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : StarRing R
        inst✝⁵ : NonUnitalSemiring A
        inst✝⁴ : StarRing A
        inst✝³ : Module R A
        inst✝² : IsScalarTower R A A
        inst✝¹ : SMulCommClass R A A
        inst✝ : StarModule R A
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalStarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (NonUnitalStarAlgebra.adjoin R s) (NonUnitalStarSubalgebra.centra …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁⟩)
      -/
      apply this at h₁
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : StarRing R
        inst✝⁵ : NonUnitalSemiring A
        inst✝⁴ : StarRing A
        inst✝³ : Module R A
        inst✝² : IsScalarTower R A A
        inst✝¹ : SMulCommClass R A A
        inst✝ : StarModule R A
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalStarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (NonUnitalStarAlgebra.adjoin R s) (NonUnitalStarSubalgebra.centra …
        h₁ : Membership.mem (NonUnitalStarSubalgebra.centralizer R ↑(NonUnitalStarSuba …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁✝⟩)
      -/
      apply this at h₂
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : StarRing R
        inst✝⁵ : NonUnitalSemiring A
        inst✝⁴ : StarRing A
        inst✝³ : Module R A
        inst✝² : IsScalarTower R A A
        inst✝¹ : SMulCommClass R A A
        inst✝ : StarModule R A
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalStarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (NonUnitalStarAlgebra.adjoin R s) (NonUnitalStarSubalgebra.centra …
        h₁ : Membership.mem (NonUnitalStarSubalgebra.centralizer R ↑(NonUnitalStarSuba …
        h₂ : Membership.mem (NonUnitalStarSubalgebra.centralizer R ↑(NonUnitalStarSuba …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂✝⟩) (HMul.hMul ⟨val✝, h₂✝⟩ ⟨val✝¹, h₁✝⟩)
      -/
      rw [← SetLike.mem_coe, coe_centralizer_centralizer] at h₁ h₂
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : StarRing R
        inst✝⁵ : NonUnitalSemiring A
        inst✝⁴ : StarRing A
        inst✝³ : Module R A
        inst✝² : IsScalarTower R A A
        inst✝¹ : SMulCommClass R A A
        inst✝ : StarModule R A
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalStarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂✝ : Membership.mem (NonUnitalStarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (NonUnitalStarAlgebra.adjoin R s) (NonUnitalStarSubalgebra.centra …
        h₁ : Membership.mem (Union.union s (Star.star s)).centralizer.centralizer val✝¹
        h₂ : Membership.mem (Union.union s (Star.star s)).centralizer.centralizer val✝
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂✝⟩) (HMul.hMul ⟨val✝, h₂✝⟩ ⟨val✝¹, h₁✝⟩)
      -/
      exact Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ h₁ _ h₂ }
      /-
        🎉 no goals
      -/


/-- If all elements of `s : Set A` commute pairwise and with elements of `star s`, then `adjoin R s`
is a non-unital commutative ring.

See note [reducible non-instances]. -/
abbrev adjoinNonUnitalCommRingOfComm (R : Type*) {A : Type*} [CommRing R] [StarRing R]
    [NonUnitalRing A] [StarRing A] [Module R A] [IsScalarTower R A A] [SMulCommClass R A A]
    [StarModule R A] {s : Set A} (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a)
    (hcomm_star : ∀ a ∈ s, ∀ b ∈ s, a * star b = star b * a) : NonUnitalCommRing (adjoin R s) :=
  { (adjoin R s).toNonUnitalRing, adjoinNonUnitalCommSemiringOfComm R hcomm hcomm_star with }


