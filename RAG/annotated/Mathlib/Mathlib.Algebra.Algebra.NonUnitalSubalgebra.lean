/-- Embedding of a non-unital subalgebra into the non-unital algebra. -/
def subtype (s : S) : s →ₙₐ[R] A :=
  { NonUnitalSubsemiringClass.subtype s, SMulMemClass.subtype s with toFun := (↑) }


@[simp]
theorem coeSubtype : (subtype s : s → A) = ((↑) : s → A) :=
  rfl


/-- A non-unital subalgebra is a sub(semi)ring that is also a submodule. -/
structure NonUnitalSubalgebra (R : Type u) (A : Type v) [CommSemiring R]
    [NonUnitalNonAssocSemiring A] [Module R A]
    extends NonUnitalSubsemiring A, Submodule R A : Type v


instance : SetLike (NonUnitalSubalgebra R A) A where
  coe s := s.carrier
                             /-
                               F : Type v'
                               R' : Type u'
                               R : Type u
                               A : Type v
                               B : Type w
                               C : Type w'
                               inst✝⁶ : CommSemiring R
                               inst✝⁵ : NonUnitalNonAssocSemiring A
                               inst✝⁴ : NonUnitalNonAssocSemiring B
                               inst✝³ : NonUnitalNonAssocSemiring C
                               inst✝² : Module R A
                               inst✝¹ : Module R B
                               inst✝ : Module R C
                               p q : NonUnitalSubalgebra R A
                               h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance instNonUnitalSubsemiringClass :
    NonUnitalSubsemiringClass (NonUnitalSubalgebra R A) A where
  add_mem {s} := s.add_mem'
  mul_mem {s} := s.mul_mem'
  zero_mem {s} := s.zero_mem'


instance instSMulMemClass : SMulMemClass (NonUnitalSubalgebra R A) R A where
  smul_mem := @fun s => s.smul_mem'


theorem mem_carrier {s : NonUnitalSubalgebra R A} {x : A} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[ext]
theorem ext {S T : NonUnitalSubalgebra R A} (h : ∀ x : A, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
theorem mem_toNonUnitalSubsemiring {S : NonUnitalSubalgebra R A} {x} :
    x ∈ S.toNonUnitalSubsemiring ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalSubsemiring (S : NonUnitalSubalgebra R A) :
    (↑S.toNonUnitalSubsemiring : Set A) = S :=
  rfl


theorem toNonUnitalSubsemiring_injective :
    Function.Injective
      (toNonUnitalSubsemiring : NonUnitalSubalgebra R A → NonUnitalSubsemiring A) :=
  fun S T h =>
                  /-
                    R : Type u
                    A : Type v
                    inst✝² : CommSemiring R
                    inst✝¹ : NonUnitalNonAssocSemiring A
                    inst✝ : Module R A
                    S T : NonUnitalSubalgebra R A
                    h : Eq S.toNonUnitalSubsemiring T.toNonUnitalSubsemiring
                    x : A
                    ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                  -/
  ext fun x => by rw [← mem_toNonUnitalSubsemiring, ← mem_toNonUnitalSubsemiring, h]
                  /-
                    🎉 no goals
                  -/


theorem toNonUnitalSubsemiring_inj {S U : NonUnitalSubalgebra R A} :
    S.toNonUnitalSubsemiring = U.toNonUnitalSubsemiring ↔ S = U :=
  toNonUnitalSubsemiring_injective.eq_iff


theorem mem_toSubmodule (S : NonUnitalSubalgebra R A) {x} : x ∈ S.toSubmodule ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toSubmodule (S : NonUnitalSubalgebra R A) : (↑S.toSubmodule : Set A) = S :=
  rfl


theorem toSubmodule_injective :
    Function.Injective (toSubmodule : NonUnitalSubalgebra R A → Submodule R A) := fun S T h =>
                  /-
                    R : Type u
                    A : Type v
                    inst✝² : CommSemiring R
                    inst✝¹ : NonUnitalNonAssocSemiring A
                    inst✝ : Module R A
                    S T : NonUnitalSubalgebra R A
                    h : Eq S.toSubmodule T.toSubmodule
                    x : A
                    ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                  -/
  ext fun x => by rw [← mem_toSubmodule, ← mem_toSubmodule, h]
                  /-
                    🎉 no goals
                  -/


theorem toSubmodule_inj {S U : NonUnitalSubalgebra R A} : S.toSubmodule = U.toSubmodule ↔ S = U :=
  toSubmodule_injective.eq_iff


/-- Copy of a non-unital subalgebra with a new `carrier` equal to the old one.
Useful to fix definitional equalities. -/
protected def copy (S : NonUnitalSubalgebra R A) (s : Set A) (hs : s = ↑S) :
    NonUnitalSubalgebra R A :=
  { S.toNonUnitalSubsemiring.copy s hs with
    smul_mem' := fun r a (ha : a ∈ s) => by
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : NonUnitalNonAssocSemiring A
        inst✝⁴ : NonUnitalNonAssocSemiring B
        inst✝³ : NonUnitalNonAssocSemiring C
        inst✝² : Module R A
        inst✝¹ : Module R B
        inst✝ : Module R C
        S : NonUnitalSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        r : R
        a : A
        ha : Membership.mem s a
        ⊢ Membership.mem __src✝.carrier (HSMul.hSMul r a)
      -/
      show r • a ∈ s
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : NonUnitalNonAssocSemiring A
        inst✝⁴ : NonUnitalNonAssocSemiring B
        inst✝³ : NonUnitalNonAssocSemiring C
        inst✝² : Module R A
        inst✝¹ : Module R B
        inst✝ : Module R C
        S : NonUnitalSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        r : R
        a : A
        ha : Membership.mem s a
        ⊢ Membership.mem s (HSMul.hSMul r a)
      -/
      rw [hs] at ha ⊢
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : NonUnitalNonAssocSemiring A
        inst✝⁴ : NonUnitalNonAssocSemiring B
        inst✝³ : NonUnitalNonAssocSemiring C
        inst✝² : Module R A
        inst✝¹ : Module R B
        inst✝ : Module R C
        S : NonUnitalSubalgebra R A
        s : Set A
        hs : Eq s ↑S
        r : R
        a : A
        ha : Membership.mem (↑S) a
        ⊢ Membership.mem (↑S) (HSMul.hSMul r a)
      -/
      exact S.smul_mem' r ha }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_copy (S : NonUnitalSubalgebra R A) (s : Set A) (hs : s = ↑S) :
    (S.copy s hs : Set A) = s :=
  rfl


theorem copy_eq (S : NonUnitalSubalgebra R A) (s : Set A) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


instance (S : NonUnitalSubalgebra R A) : Inhabited S :=
  ⟨(0 : S.toNonUnitalSubsemiring)⟩


instance instNonUnitalSubringClass : NonUnitalSubringClass (NonUnitalSubalgebra R A) A :=
  { NonUnitalSubalgebra.instNonUnitalSubsemiringClass with
    neg_mem := @fun _ x hx => neg_one_smul R x ▸ SMulMemClass.smul_mem _ hx }


/-- A non-unital subalgebra over a ring is also a `Subring`. -/
def toNonUnitalSubring (S : NonUnitalSubalgebra R A) : NonUnitalSubring A where
  toNonUnitalSubsemiring := S.toNonUnitalSubsemiring
  neg_mem' := neg_mem (s := S)


@[simp]
theorem mem_toNonUnitalSubring {S : NonUnitalSubalgebra R A} {x} :
    x ∈ S.toNonUnitalSubring ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalSubring (S : NonUnitalSubalgebra R A) :
    (↑S.toNonUnitalSubring : Set A) = S :=
  rfl


theorem toNonUnitalSubring_injective :
    Function.Injective (toNonUnitalSubring : NonUnitalSubalgebra R A → NonUnitalSubring A) :=
                               /-
                                 R : Type u
                                 A : Type v
                                 inst✝² : CommRing R
                                 inst✝¹ : NonUnitalNonAssocRing A
                                 inst✝ : Module R A
                                 S T : NonUnitalSubalgebra R A
                                 h : Eq S.toNonUnitalSubring T.toNonUnitalSubring
                                 x : A
                                 ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                               -/
  fun S T h => ext fun x => by rw [← mem_toNonUnitalSubring, ← mem_toNonUnitalSubring, h]
                               /-
                                 🎉 no goals
                               -/


theorem toNonUnitalSubring_inj {S U : NonUnitalSubalgebra R A} :
    S.toNonUnitalSubring = U.toNonUnitalSubring ↔ S = U :=
  toNonUnitalSubring_injective.eq_iff


instance toNonUnitalNonAssocSemiring [CommSemiring R] [NonUnitalNonAssocSemiring A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalNonAssocSemiring S :=
  inferInstance


instance toNonUnitalSemiring [CommSemiring R] [NonUnitalSemiring A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalSemiring S :=
  inferInstance


instance toNonUnitalCommSemiring [CommSemiring R] [NonUnitalCommSemiring A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalCommSemiring S :=
  inferInstance


instance toNonUnitalNonAssocRing [CommRing R] [NonUnitalNonAssocRing A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalNonAssocRing S :=
  inferInstance


instance toNonUnitalRing [CommRing R] [NonUnitalRing A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalRing S :=
  inferInstance


instance toNonUnitalCommRing [CommRing R] [NonUnitalCommRing A] [Module R A]
    (S : NonUnitalSubalgebra R A) : NonUnitalCommRing S :=
  inferInstance


/-- The forgetful map from `NonUnitalSubalgebra` to `Submodule` as an `OrderEmbedding` -/
def toSubmodule' [CommSemiring R] [NonUnitalNonAssocSemiring A] [Module R A] :
    NonUnitalSubalgebra R A ↪o Submodule R A where
  toEmbedding :=
    { toFun := fun S => S.toSubmodule
                                     /-
                                       F : Type v'
                                       R' : Type u'
                                       R : Type u
                                       A : Type v
                                       B : Type w
                                       C : Type w'
                                       inst✝² : CommSemiring R
                                       inst✝¹ : NonUnitalNonAssocSemiring A
                                       inst✝ : Module R A
                                       S T : NonUnitalSubalgebra R A
                                       h : Eq ((fun S => S.toSubmodule) S) ((fun S => S.toSubmodule) T)
                                       ⊢ ∀ (x : A), Iff (Membership.mem S x) (Membership.mem T x)
                                     -/
      inj' := fun S T h => ext <| by apply SetLike.ext_iff.1 h }
                                     /-
                                       🎉 no goals
                                     -/
  map_rel_iff' := SetLike.coe_subset_coe.symm.trans SetLike.coe_subset_coe


/-- The forgetful map from `NonUnitalSubalgebra` to `NonUnitalSubsemiring` as an
`OrderEmbedding` -/
def toNonUnitalSubsemiring' [CommSemiring R] [NonUnitalNonAssocSemiring A] [Module R A] :
    NonUnitalSubalgebra R A ↪o NonUnitalSubsemiring A where
  toEmbedding :=
    { toFun := fun S => S.toNonUnitalSubsemiring
                                     /-
                                       F : Type v'
                                       R' : Type u'
                                       R : Type u
                                       A : Type v
                                       B : Type w
                                       C : Type w'
                                       inst✝² : CommSemiring R
                                       inst✝¹ : NonUnitalNonAssocSemiring A
                                       inst✝ : Module R A
                                       S T : NonUnitalSubalgebra R A
                                       h : Eq ((fun S => S.toNonUnitalSubsemiring) S) ((fun S => S.toNonUnitalSubsemi …
                                       ⊢ ∀ (x : A), Iff (Membership.mem S x) (Membership.mem T x)
                                     -/
      inj' := fun S T h => ext <| by apply SetLike.ext_iff.1 h }
                                     /-
                                       🎉 no goals
                                     -/
  map_rel_iff' := SetLike.coe_subset_coe.symm.trans SetLike.coe_subset_coe


/-- The forgetful map from `NonUnitalSubalgebra` to `NonUnitalSubsemiring` as an
`OrderEmbedding` -/
def toNonUnitalSubring' [CommRing R] [NonUnitalNonAssocRing A] [Module R A] :
    NonUnitalSubalgebra R A ↪o NonUnitalSubring A where
  toEmbedding :=
    { toFun := fun S => S.toNonUnitalSubring
                                     /-
                                       F : Type v'
                                       R' : Type u'
                                       R : Type u
                                       A : Type v
                                       B : Type w
                                       C : Type w'
                                       inst✝² : CommRing R
                                       inst✝¹ : NonUnitalNonAssocRing A
                                       inst✝ : Module R A
                                       S T : NonUnitalSubalgebra R A
                                       h : Eq ((fun S => S.toNonUnitalSubring) S) ((fun S => S.toNonUnitalSubring) T)
                                       ⊢ ∀ (x : A), Iff (Membership.mem S x) (Membership.mem T x)
                                     -/
      inj' := fun S T h => ext <| by apply SetLike.ext_iff.1 h }
                                     /-
                                       🎉 no goals
                                     -/
  map_rel_iff' := SetLike.coe_subset_coe.symm.trans SetLike.coe_subset_coe


instance instModule' [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] : Module R' S :=
  SMulMemClass.toModule' _ R' R A S


instance instModule : Module R S :=
  S.instModule'


instance instIsScalarTower' [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] :
    IsScalarTower R' R S :=
  S.toSubmodule.isScalarTower


instance [IsScalarTower R A A] : IsScalarTower R S S where
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


protected theorem coe_neg {R : Type u} {A : Type v} [CommRing R] [Ring A] [Algebra R A]
    {S : NonUnitalSubalgebra R A} (x : S) : (↑(-x) : A) = -↑x :=
  rfl


protected theorem coe_sub {R : Type u} {A : Type v} [CommRing R] [Ring A] [Algebra R A]
    {S : NonUnitalSubalgebra R A} (x y : S) : (↑(x - y) : A) = ↑x - ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_smul [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A] (r : R') (x : S) :
    ↑(r • x) = r • (x : A) :=
  rfl


protected theorem coe_eq_zero {x : S} : (x : A) = 0 ↔ x = 0 :=
  ZeroMemClass.coe_eq_zero


@[simp]
theorem toNonUnitalSubsemiring_subtype :
    NonUnitalSubsemiringClass.subtype S = NonUnitalSubalgebraClass.subtype (R := R) S :=
  rfl


@[simp]
theorem toSubring_subtype {R A : Type*} [CommRing R] [Ring A] [Algebra R A]
    (S : NonUnitalSubalgebra R A) :
    NonUnitalSubringClass.subtype S = NonUnitalSubalgebraClass.subtype (R := R) S :=
  rfl


/-- Linear equivalence between `S : Submodule R A` and `S`. Though these types are equal,
we define it as a `LinearEquiv` to avoid type equalities. -/
def toSubmoduleEquiv (S : NonUnitalSubalgebra R A) : S.toSubmodule ≃ₗ[R] S :=
  LinearEquiv.ofEq _ _ rfl


/-- Transport a non-unital subalgebra via an algebra homomorphism. -/
def map (f : F) (S : NonUnitalSubalgebra R A) : NonUnitalSubalgebra R B :=
  { S.toNonUnitalSubsemiring.map (f : A →ₙ+* B) with
    smul_mem' := fun r b hb => by
      /-
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁸ : CommSemiring R
        inst✝⁷ : NonUnitalNonAssocSemiring A
        inst✝⁶ : NonUnitalNonAssocSemiring B
        inst✝⁵ : NonUnitalNonAssocSemiring C
        inst✝⁴ : Module R A
        inst✝³ : Module R B
        inst✝² : Module R C
        S✝ : NonUnitalSubalgebra R A
        inst✝¹ : FunLike F A B
        inst✝ : NonUnitalAlgHomClass F R A B
        f : F
        S : NonUnitalSubalgebra R A
        r : R
        b : B
        hb : Membership.mem __src✝.carrier b
        ⊢ Membership.mem __src✝.carrier (HSMul.hSMul r b)
      -/
      rcases hb with ⟨a, ha, rfl⟩
      /-
        case intro.intro
        F : Type v'
        R' : Type u'
        R : Type u
        A : Type v
        B : Type w
        C : Type w'
        inst✝⁸ : CommSemiring R
        inst✝⁷ : NonUnitalNonAssocSemiring A
        inst✝⁶ : NonUnitalNonAssocSemiring B
        inst✝⁵ : NonUnitalNonAssocSemiring C
        inst✝⁴ : Module R A
        inst✝³ : Module R B
        inst✝² : Module R C
        S✝ : NonUnitalSubalgebra R A
        inst✝¹ : FunLike F A B
        inst✝ : NonUnitalAlgHomClass F R A B
        f : F
        S : NonUnitalSubalgebra R A
        r : R
        a : A
        ha : Membership.mem (↑S.toNonUnitalSubsemiring) a
        ⊢ Membership.mem __src✝.carrier (HSMul.hSMul r (↑f a))
      -/
      exact map_smulₛₗ f r a ▸ Set.mem_image_of_mem f (S.smul_mem' r ha) }
      /-
        🎉 no goals
      -/


theorem map_mono {S₁ S₂ : NonUnitalSubalgebra R A} {f : F} :
    S₁ ≤ S₂ → (map f S₁ : NonUnitalSubalgebra R B) ≤ map f S₂ :=
  Set.image_subset f


theorem map_injective {f : F} (hf : Function.Injective f) :
    Function.Injective (map f : NonUnitalSubalgebra R A → NonUnitalSubalgebra R B) :=
  fun _S₁ _S₂ ih =>
  ext <| Set.ext_iff.1 <| Set.image_injective.2 hf <| Set.ext <| SetLike.ext_iff.mp ih


@[simp]
theorem map_id (S : NonUnitalSubalgebra R A) : map (NonUnitalAlgHom.id R A) S = S :=
  SetLike.coe_injective <| Set.image_id _


theorem map_map (S : NonUnitalSubalgebra R A) (g : B →ₙₐ[R] C) (f : A →ₙₐ[R] B) :
    (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image _ _ _


@[simp]
theorem mem_map {S : NonUnitalSubalgebra R A} {f : F} {y : B} : y ∈ map f S ↔ ∃ x ∈ S, f x = y :=
  NonUnitalSubsemiring.mem_map


theorem map_toSubmodule {S : NonUnitalSubalgebra R A} {f : F} :
    -- TODO: introduce a better coercion from `NonUnitalAlgHomClass` to `LinearMap`
    (map f S).toSubmodule = Submodule.map (LinearMapClass.linearMap f) S.toSubmodule :=
  SetLike.coe_injective rfl


theorem map_toNonUnitalSubsemiring {S : NonUnitalSubalgebra R A} {f : F} :
    (map f S).toNonUnitalSubsemiring = S.toNonUnitalSubsemiring.map (f : A →ₙ+* B) :=
  SetLike.coe_injective rfl


@[simp]
theorem coe_map (S : NonUnitalSubalgebra R A) (f : F) : (map f S : Set B) = f '' S :=
  rfl


/-- Preimage of a non-unital subalgebra under an algebra homomorphism. -/
def comap (f : F) (S : NonUnitalSubalgebra R B) : NonUnitalSubalgebra R A :=
  { S.toNonUnitalSubsemiring.comap (f : A →ₙ+* B) with
    smul_mem' := fun r a (ha : f a ∈ S) =>
      show f (r • a) ∈ S from (map_smulₛₗ f r a).symm ▸ SMulMemClass.smul_mem r ha }


theorem map_le {S : NonUnitalSubalgebra R A} {f : F} {U : NonUnitalSubalgebra R B} :
    map f S ≤ U ↔ S ≤ comap f U :=
  Set.image_subset_iff


theorem gc_map_comap (f : F) :
    GaloisConnection (map f : NonUnitalSubalgebra R A → NonUnitalSubalgebra R B) (comap f) :=
  fun _ _ => map_le


@[simp]
theorem mem_comap (S : NonUnitalSubalgebra R B) (f : F) (x : A) : x ∈ comap f S ↔ f x ∈ S :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_comap (S : NonUnitalSubalgebra R B) (f : F) : (comap f S : Set A) = f ⁻¹' (S : Set B) :=
  rfl


instance noZeroDivisors {R A : Type*} [CommSemiring R] [NonUnitalSemiring A] [NoZeroDivisors A]
    [Module R A] (S : NonUnitalSubalgebra R A) : NoZeroDivisors S :=
  NonUnitalSubsemiringClass.noZeroDivisors S


/-- A submodule closed under multiplication is a non-unital subalgebra. -/
def toNonUnitalSubalgebra (p : Submodule R A) (h_mul : ∀ x y, x ∈ p → y ∈ p → x * y ∈ p) :
    NonUnitalSubalgebra R A :=
  { p with
    mul_mem' := h_mul _ _ }


@[simp]
theorem mem_toNonUnitalSubalgebra {p : Submodule R A} {h_mul} {x} :
    x ∈ p.toNonUnitalSubalgebra h_mul ↔ x ∈ p :=
  Iff.rfl


@[simp]
theorem coe_toNonUnitalSubalgebra (p : Submodule R A) (h_mul) :
    (p.toNonUnitalSubalgebra h_mul : Set A) = p :=
  rfl


theorem toNonUnitalSubalgebra_mk (p : Submodule R A) hmul :
    p.toNonUnitalSubalgebra hmul =
      NonUnitalSubalgebra.mk ⟨⟨⟨p, p.add_mem⟩, p.zero_mem⟩, hmul _ _⟩ p.smul_mem' :=
  rfl


@[simp]
theorem toNonUnitalSubalgebra_toSubmodule (p : Submodule R A) (h_mul) :
    (p.toNonUnitalSubalgebra h_mul).toSubmodule = p :=
  SetLike.coe_injective rfl


@[simp]
theorem _root_.NonUnitalSubalgebra.toSubmodule_toNonUnitalSubalgebra (S : NonUnitalSubalgebra R A) :
    (S.toSubmodule.toNonUnitalSubalgebra fun _ _ => mul_mem (s := S)) = S :=
  SetLike.coe_injective rfl


/-- Range of an `NonUnitalAlgHom` as a non-unital subalgebra. -/
protected def range (φ : F) : NonUnitalSubalgebra R B where
  toNonUnitalSubsemiring := NonUnitalRingHom.srange (φ : A →ₙ+* B)
                             /-
                               F : Type v'
                               R' : Type u'
                               R : Type u
                               A : Type v
                               B : Type w
                               C : Type w'
                               inst✝⁸ : CommSemiring R
                               inst✝⁷ : NonUnitalNonAssocSemiring A
                               inst✝⁶ : Module R A
                               inst✝⁵ : NonUnitalNonAssocSemiring B
                               inst✝⁴ : Module R B
                               inst✝³ : NonUnitalNonAssocSemiring C
                               inst✝² : Module R C
                               inst✝¹ : FunLike F A B
                               inst✝ : NonUnitalAlgHomClass F R A B
                               φ : F
                               r : R
                               a : B
                               ⊢ Membership.mem (NonUnitalRingHom.srange ↑φ).carrier a → Membership.mem (NonU …
                             -/
  smul_mem' := fun r a => by rintro ⟨a, rfl⟩; exact ⟨r • a, map_smul φ r a⟩
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem mem_range (φ : F) {y : B} :
    y ∈ (NonUnitalAlgHom.range φ : NonUnitalSubalgebra R B) ↔ ∃ x : A, φ x = y :=
  NonUnitalRingHom.mem_srange


theorem mem_range_self (φ : F) (x : A) :
    φ x ∈ (NonUnitalAlgHom.range φ : NonUnitalSubalgebra R B) :=
  (NonUnitalAlgHom.mem_range φ).2 ⟨x, rfl⟩


@[simp]
theorem coe_range (φ : F) :
    ((NonUnitalAlgHom.range φ : NonUnitalSubalgebra R B) : Set B) = Set.range (φ : A → B) := by
  /-
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : NonUnitalNonAssocSemiring A
    inst✝⁴ : Module R A
    inst✝³ : NonUnitalNonAssocSemiring B
    inst✝² : Module R B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F R A B
    φ : F
    ⊢ Eq (↑(NonUnitalAlgHom.range φ)) (Set.range ⇑φ)
  -/
  ext
  /-
    case h
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : NonUnitalNonAssocSemiring A
    inst✝⁴ : Module R A
    inst✝³ : NonUnitalNonAssocSemiring B
    inst✝² : Module R B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F R A B
    φ : F
    x✝ : B
    ⊢ Iff (Membership.mem (↑(NonUnitalAlgHom.range φ)) x✝) (Membership.mem (Set.ra …
  -/
  rw [SetLike.mem_coe, mem_range]
  /-
    case h
    F : Type v'
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : NonUnitalNonAssocSemiring A
    inst✝⁴ : Module R A
    inst✝³ : NonUnitalNonAssocSemiring B
    inst✝² : Module R B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F R A B
    φ : F
    x✝ : B
    ⊢ Iff (Exists fun x => Eq (φ x) x✝) (Membership.mem (Set.range ⇑φ) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem range_comp (f : A →ₙₐ[R] B) (g : B →ₙₐ[R] C) :
    NonUnitalAlgHom.range (g.comp f) = (NonUnitalAlgHom.range f).map g :=
  SetLike.coe_injective (Set.range_comp g f)


theorem range_comp_le_range (f : A →ₙₐ[R] B) (g : B →ₙₐ[R] C) :
    NonUnitalAlgHom.range (g.comp f) ≤ NonUnitalAlgHom.range g :=
  SetLike.coe_mono (Set.range_comp_subset_range f g)


/-- Restrict the codomain of a non-unital algebra homomorphism. -/
def codRestrict (f : F) (S : NonUnitalSubalgebra R B) (hf : ∀ x, f x ∈ S) : A →ₙₐ[R] S :=
  { NonUnitalRingHom.codRestrict (f : A →ₙ+* B) S.toNonUnitalSubsemiring hf with
    map_smul' := fun r a => Subtype.ext <| map_smul f r a }


@[simp]
theorem subtype_comp_codRestrict (f : F) (S : NonUnitalSubalgebra R B) (hf : ∀ x : A, f x ∈ S) :
    (NonUnitalSubalgebraClass.subtype S).comp (NonUnitalAlgHom.codRestrict f S hf) = f :=
  rfl


@[simp]
theorem coe_codRestrict (f : F) (S : NonUnitalSubalgebra R B) (hf : ∀ x, f x ∈ S) (x : A) :
    ↑(NonUnitalAlgHom.codRestrict f S hf x) = f x :=
  rfl


theorem injective_codRestrict (f : F) (S : NonUnitalSubalgebra R B) (hf : ∀ x : A, f x ∈ S) :
    Function.Injective (NonUnitalAlgHom.codRestrict f S hf) ↔ Function.Injective f :=
  ⟨fun H _x _y hxy => H <| Subtype.eq hxy, fun H _x _y hxy => H (congr_arg Subtype.val hxy : _)⟩


/-- Restrict the codomain of an `NonUnitalAlgHom` `f` to `f.range`.

This is the bundled version of `Set.rangeFactorization`. -/
abbrev rangeRestrict (f : F) : A →ₙₐ[R] (NonUnitalAlgHom.range f : NonUnitalSubalgebra R B) :=
  NonUnitalAlgHom.codRestrict f (NonUnitalAlgHom.range f) (NonUnitalAlgHom.mem_range_self f)


/-- The equalizer of two non-unital `R`-algebra homomorphisms -/
def equalizer (ϕ ψ : F) : NonUnitalSubalgebra R A where
  carrier := {a | (ϕ a : B) = ψ a}
                  /-
                    F : Type v'
                    R' : Type u'
                    R : Type u
                    A : Type v
                    B : Type w
                    C : Type w'
                    inst✝⁸ : CommSemiring R
                    inst✝⁷ : NonUnitalNonAssocSemiring A
                    inst✝⁶ : Module R A
                    inst✝⁵ : NonUnitalNonAssocSemiring B
                    inst✝⁴ : Module R B
                    inst✝³ : NonUnitalNonAssocSemiring C
                    inst✝² : Module R C
                    inst✝¹ : FunLike F A B
                    inst✝ : NonUnitalAlgHomClass F R A B
                    ϕ ψ : F
                    ⊢ Membership.mem { carrier := setOf fun a => Eq (ϕ a) (ψ a), add_mem' := ⋯ }.c …
                  -/
  zero_mem' := by rw [Set.mem_setOf_eq, map_zero, map_zero]
    /-
      F : Type v'
      R' : Type u'
      R : Type u
      A : Type v
      B : Type w
      C : Type w'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : NonUnitalNonAssocSemiring A
      inst✝⁶ : Module R A
      inst✝⁵ : NonUnitalNonAssocSemiring B
      inst✝⁴ : Module R B
      inst✝³ : NonUnitalNonAssocSemiring C
      inst✝² : Module R C
      inst✝¹ : FunLike F A B
      inst✝ : NonUnitalAlgHomClass F R A B
      ϕ ψ : F
      x y : A
      hx : Eq (ϕ x) (ψ x)
      hy : Eq (ϕ y) (ψ y)
      ⊢ Membership.mem (setOf fun a => Eq (ϕ a) (ψ a)) (HAdd.hAdd x y)
    -/
                  /-
                    🎉 no goals
                  -/
    /-
      🎉 no goals
    -/
  add_mem' {x y} (hx : ϕ x = ψ x) (hy : ϕ y = ψ y) := by
    rw [Set.mem_setOf_eq, map_add, map_add, hx, hy]
  mul_mem' {x y} (hx : ϕ x = ψ x) (hy : ϕ y = ψ y) := by
    /-
      F : Type v'
      R' : Type u'
      R : Type u
      A : Type v
      B : Type w
      C : Type w'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : NonUnitalNonAssocSemiring A
      inst✝⁶ : Module R A
      inst✝⁵ : NonUnitalNonAssocSemiring B
      inst✝⁴ : Module R B
      inst✝³ : NonUnitalNonAssocSemiring C
      inst✝² : Module R C
      inst✝¹ : FunLike F A B
      inst✝ : NonUnitalAlgHomClass F R A B
      ϕ ψ : F
      x y : A
      hx : Eq (ϕ x) (ψ x)
      hy : Eq (ϕ y) (ψ y)
      ⊢ Membership.mem { carrier := setOf fun a => Eq (ϕ a) (ψ a), add_mem' := ⋯, ze …
    -/
    rw [Set.mem_setOf_eq, map_mul, map_mul, hx, hy]
    /-
      🎉 no goals
    -/
                                       /-
                                         F : Type v'
                                         R' : Type u'
                                         R : Type u
                                         A : Type v
                                         B : Type w
                                         C : Type w'
                                         inst✝⁸ : CommSemiring R
                                         inst✝⁷ : NonUnitalNonAssocSemiring A
                                         inst✝⁶ : Module R A
                                         inst✝⁵ : NonUnitalNonAssocSemiring B
                                         inst✝⁴ : Module R B
                                         inst✝³ : NonUnitalNonAssocSemiring C
                                         inst✝² : Module R C
                                         inst✝¹ : FunLike F A B
                                         inst✝ : NonUnitalAlgHomClass F R A B
                                         ϕ ψ : F
                                         r : R
                                         x : A
                                         hx : Eq (ϕ x) (ψ x)
                                         ⊢ Membership.mem { carrier := setOf fun a => Eq (ϕ a) (ψ a), add_mem' := ⋯, ze …
                                       -/
  smul_mem' r x (hx : ϕ x = ψ x) := by rw [Set.mem_setOf_eq, map_smul, map_smul, hx]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem mem_equalizer (φ ψ : F) (x : A) :
    x ∈ NonUnitalAlgHom.equalizer φ ψ ↔ φ x = ψ x :=
  Iff.rfl


/-- The range of a morphism of algebras is a fintype, if the domain is a fintype.

Note that this instance can cause a diamond with `Subtype.fintype` if `B` is also a fintype. -/
instance fintypeRange [Fintype A] [DecidableEq B] (φ : F) :
    Fintype (NonUnitalAlgHom.range φ) :=
  Set.fintypeRange φ


@[simp]
lemma span_eq_toSubmodule (s : NonUnitalSubalgebra R A) :
    Submodule.span R (s : Set A) = s.toSubmodule := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : NonUnitalNonAssocSemiring A
    inst✝ : Module R A
    s : NonUnitalSubalgebra R A
    ⊢ Eq (Submodule.span R ↑s) s.toSubmodule
  -/
  simp [SetLike.ext'_iff, Submodule.coe_span_eq_self]
  /-
    🎉 no goals
  -/


/-- The minimal non-unital subalgebra that includes `s`. -/
def adjoin (s : Set A) : NonUnitalSubalgebra R A :=
  { Submodule.span R (NonUnitalSubsemiring.closure s : Set A) with
    mul_mem' :=
      @fun a b (ha : a ∈ Submodule.span R (NonUnitalSubsemiring.closure s : Set A))
        (hb : b ∈ Submodule.span R (NonUnitalSubsemiring.closure s : Set A)) =>
      show a * b ∈ Submodule.span R (NonUnitalSubsemiring.closure s : Set A) by
        /-
          F : Type u_1
          R : Type u
          A : Type v
          B : Type w
          inst✝⁸ : CommSemiring R
          inst✝⁷ : NonUnitalNonAssocSemiring A
          inst✝⁶ : Module R A
          inst✝⁵ : NonUnitalNonAssocSemiring B
          inst✝⁴ : Module R B
          inst✝³ : FunLike F A B
          inst✝² : NonUnitalAlgHomClass F R A B
          inst✝¹ : IsScalarTower R A A
          inst✝ : SMulCommClass R A A
          s : Set A
          a b : A
          ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
          hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
          ⊢ Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) (HMul.hM …
        -/
        refine Submodule.span_induction ?_ ?_ ?_ ?_ ha
          /-
            case refine_1
            F : Type u_1
            R : Type u
            A : Type v
            B : Type w
            inst✝⁸ : CommSemiring R
            inst✝⁷ : NonUnitalNonAssocSemiring A
            inst✝⁶ : Module R A
            inst✝⁵ : NonUnitalNonAssocSemiring B
            inst✝⁴ : Module R B
            inst✝³ : FunLike F A B
            inst✝² : NonUnitalAlgHomClass F R A B
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            s : Set A
            a b : A
            ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
            hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
            ⊢ ∀ (x : A), Membership.mem (↑(NonUnitalSubsemiring.closure s)) x → Membership …
          -/
        · refine Submodule.span_induction ?_ ?_ ?_ ?_ hb
          · exact fun x (hx : x ∈ NonUnitalSubsemiring.closure s) y
              (hy : y ∈ NonUnitalSubsemiring.closure s) => Submodule.subset_span (mul_mem hy hx)
            /-
              case refine_1.refine_2
              F : Type u_1
              R : Type u
              A : Type v
              B : Type w
              inst✝⁸ : CommSemiring R
              inst✝⁷ : NonUnitalNonAssocSemiring A
              inst✝⁶ : Module R A
              inst✝⁵ : NonUnitalNonAssocSemiring B
              inst✝⁴ : Module R B
              inst✝³ : FunLike F A B
              inst✝² : NonUnitalAlgHomClass F R A B
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              s : Set A
              a b : A
              ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
              hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
              ⊢ ∀ (x : A), Membership.mem (↑(NonUnitalSubsemiring.closure s)) x → Membership …
            -/
          · exact fun x _hx => (mul_zero x).symm ▸ Submodule.zero_mem _
            /-
              🎉 no goals
            -/
            /-
              case refine_1.refine_3
              F : Type u_1
              R : Type u
              A : Type v
              B : Type w
              inst✝⁸ : CommSemiring R
              inst✝⁷ : NonUnitalNonAssocSemiring A
              inst✝⁶ : Module R A
              inst✝⁵ : NonUnitalNonAssocSemiring B
              inst✝⁴ : Module R B
              inst✝³ : FunLike F A B
              inst✝² : NonUnitalAlgHomClass F R A B
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              s : Set A
              a b : A
              ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
              hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
              ⊢ ∀ (x y : A), Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure …
            -/
          · exact fun x y _ _ hx hy z hz => (mul_add z x y).symm ▸ add_mem (hx z hz) (hy z hz)
            /-
              🎉 no goals
            -/
          · exact fun r x _ hx y hy =>
              (mul_smul_comm r y x).symm ▸ SMulMemClass.smul_mem r (hx y hy)
          /-
            case refine_2
            F : Type u_1
            R : Type u
            A : Type v
            B : Type w
            inst✝⁸ : CommSemiring R
            inst✝⁷ : NonUnitalNonAssocSemiring A
            inst✝⁶ : Module R A
            inst✝⁵ : NonUnitalNonAssocSemiring B
            inst✝⁴ : Module R B
            inst✝³ : FunLike F A B
            inst✝² : NonUnitalAlgHomClass F R A B
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            s : Set A
            a b : A
            ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
            hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
            ⊢ Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) (HMul.hM …
          -/
        · exact (zero_mul b).symm ▸ Submodule.zero_mem _
          /-
            🎉 no goals
          -/
          /-
            case refine_3
            F : Type u_1
            R : Type u
            A : Type v
            B : Type w
            inst✝⁸ : CommSemiring R
            inst✝⁷ : NonUnitalNonAssocSemiring A
            inst✝⁶ : Module R A
            inst✝⁵ : NonUnitalNonAssocSemiring B
            inst✝⁴ : Module R B
            inst✝³ : FunLike F A B
            inst✝² : NonUnitalAlgHomClass F R A B
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            s : Set A
            a b : A
            ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
            hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
            ⊢ ∀ (x y : A), Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure …
          -/
        · exact fun x y _ _ => (add_mul x y b).symm ▸ add_mem
          /-
            🎉 no goals
          -/
          /-
            case refine_4
            F : Type u_1
            R : Type u
            A : Type v
            B : Type w
            inst✝⁸ : CommSemiring R
            inst✝⁷ : NonUnitalNonAssocSemiring A
            inst✝⁶ : Module R A
            inst✝⁵ : NonUnitalNonAssocSemiring B
            inst✝⁴ : Module R B
            inst✝³ : FunLike F A B
            inst✝² : NonUnitalAlgHomClass F R A B
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            s : Set A
            a b : A
            ha : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) a
            hb : Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.closure s)) b
            ⊢ ∀ (a : R) (x : A), Membership.mem (Submodule.span R ↑(NonUnitalSubsemiring.c …
          -/
        · exact fun r x _ hx => (smul_mul_assoc r x b).symm ▸ SMulMemClass.smul_mem r hx }
          /-
            🎉 no goals
          -/


theorem adjoin_toSubmodule (s : Set A) :
    (adjoin R s).toSubmodule = Submodule.span R (NonUnitalSubsemiring.closure s : Set A) :=
  rfl


@[aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_adjoin {s : Set A} : s ⊆ adjoin R s :=
  NonUnitalSubsemiring.subset_closure.trans Submodule.subset_span


theorem self_mem_adjoin_singleton (x : A) : x ∈ adjoin R ({x} : Set A) :=
  NonUnitalAlgebra.subset_adjoin R (Set.mem_singleton x)


protected theorem gc : GaloisConnection (adjoin R : Set A → NonUnitalSubalgebra R A) (↑) :=
  fun s S =>
  ⟨fun H => (NonUnitalSubsemiring.subset_closure.trans Submodule.subset_span).trans H,
    fun H => show Submodule.span R _ ≤ S.toSubmodule from Submodule.span_le.mpr <|
      show NonUnitalSubsemiring.closure s ≤ S.toNonUnitalSubsemiring from
        NonUnitalSubsemiring.closure_le.2 H⟩


/-- Galois insertion between `adjoin` and `Subtype.val`. -/
protected def gi : GaloisInsertion (adjoin R : Set A → NonUnitalSubalgebra R A) (↑) where
  choice s hs := (adjoin R s).copy s <| le_antisymm (NonUnitalAlgebra.gc.le_u_l s) hs
  gc := NonUnitalAlgebra.gc
  le_l_u S := (NonUnitalAlgebra.gc (S : Set A) (adjoin R S)).1 <| le_rfl
  choice_eq _ _ := NonUnitalSubalgebra.copy_eq _ _ _


instance : CompleteLattice (NonUnitalSubalgebra R A) :=
  GaloisInsertion.liftCompleteLattice NonUnitalAlgebra.gi


theorem adjoin_le {S : NonUnitalSubalgebra R A} {s : Set A} (hs : s ⊆ S) : adjoin R s ≤ S :=
  NonUnitalAlgebra.gc.l_le hs


theorem adjoin_le_iff {S : NonUnitalSubalgebra R A} {s : Set A} : adjoin R s ≤ S ↔ s ⊆ S :=
  NonUnitalAlgebra.gc _ _


theorem adjoin_union (s t : Set A) : adjoin R (s ∪ t) = adjoin R s ⊔ adjoin R t :=
  (NonUnitalAlgebra.gc : GaloisConnection _ ((↑) : NonUnitalSubalgebra R A → Set A)).l_sup


lemma adjoin_eq (s : NonUnitalSubalgebra R A) : adjoin R (s : Set A) = s :=
  le_antisymm (adjoin_le le_rfl) (subset_adjoin R)


/-- If some predicate holds for all `x ∈ (s : Set A)` and this predicate is closed under the
`algebraMap`, addition, multiplication and star operations, then it holds for `a ∈ adjoin R s`. -/
@[elab_as_elim]
theorem adjoin_induction {s : Set A} {p : (x : A) → x ∈ adjoin R s → Prop}
    (mem : ∀ (x) (hx : x ∈ s), p x (subset_adjoin R hx))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy)) (zero : p 0 (zero_mem _))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    (smul : ∀ r x hx, p x hx → p (r • x) (SMulMemClass.smul_mem r hx))
    {x} (hx : x ∈ adjoin R s) : p x hx :=
  let S : NonUnitalSubalgebra R A :=
    { carrier := { x | ∃ hx, p x hx }
      mul_mem' := (Exists.elim · fun _ ha ↦ (Exists.elim · fun _ hb ↦ ⟨_, mul _ _ _ _ ha hb⟩))
      add_mem' := (Exists.elim · fun _ ha ↦ (Exists.elim · fun _ hb ↦ ⟨_, add _ _ _ _ ha hb⟩))
      smul_mem' := fun r ↦ (Exists.elim · fun _ hb ↦ ⟨_, smul r _ _ hb⟩)
      zero_mem' := ⟨_, zero⟩ }
  adjoin_le (S := S) (fun y hy ↦ ⟨subset_adjoin R hy, mem y hy⟩) hx |>.elim fun _ ↦ id


@[deprecated adjoin_induction (since := "2024-10-10")]
alias adjoin_induction' := adjoin_induction


@[elab_as_elim]
theorem adjoin_induction₂ {s : Set A} {p : ∀ x y, x ∈ adjoin R s → y ∈ adjoin R s → Prop}
    (mem_mem : ∀ (x) (y) (hx : x ∈ s) (hy : y ∈ s), p x y (subset_adjoin R hx) (subset_adjoin R hy))
    (zero_left : ∀ x hx, p 0 x (zero_mem _) hx) (zero_right : ∀ x hx, p x 0 hx (zero_mem _))
    (add_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x + y) z (add_mem hx hy) hz)
    (add_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y + z) hx (add_mem hy hz))
    (mul_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x * y) z (mul_mem hx hy) hz)
    (mul_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y * z) hx (mul_mem hy hz))
    (smul_left : ∀ r x y hx hy, p x y hx hy → p (r • x) y (SMulMemClass.smul_mem r hx) hy)
    (smul_right : ∀ r x y hx hy, p x y hx hy → p x (r • y) hx (SMulMemClass.smul_mem r hy))
    {x y : A} (hx : x ∈ adjoin R s) (hy : y ∈ adjoin R s) :
    p x y hx hy := by
  induction hy using adjoin_induction with
  | mem z hz =>
    induction hx using adjoin_induction with
    | mem _ h => exact mem_mem _ _ h hz
    | zero => exact zero_left _ _
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
    | smul _ _ _ h => exact smul_left _ _ _ _ _ h
  | zero => exact zero_right x hx
  | mul _ _ _ _ h₁ h₂ => exact mul_right _ _ _ _ _ _ h₁ h₂
  | add _ _ _ _ h₁ h₂ => exact add_right _ _ _ _ _ _ h₁ h₂
  | smul _ _ _ h => exact smul_right _ _ _ _ _ h


/-- The difference with `NonUnitalAlgebra.adjoin_induction` is that this acts on the subtype. -/
@[elab_as_elim, deprecated adjoin_induction (since := "2024-10-11")]
lemma adjoin_induction_subtype {s : Set A} {p : adjoin R s → Prop} (a : adjoin R s)
    (mem : ∀ x (h : x ∈ s), p ⟨x, subset_adjoin R h⟩)
    (add : ∀ x y, p x → p y → p (x + y)) (zero : p 0)
    (mul : ∀ x y, p x → p y → p (x * y)) (smul : ∀ (r : R) x, p x → p (r • x)) : p a :=
  Subtype.recOn a fun b hb => by
    induction hb using adjoin_induction with
    | mem _ h => exact mem _ h
    | zero => exact zero
    | mul _ _ _ _ h₁ h₂ => exact mul _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add _ _ h₁ h₂
    | smul _ _ _ h => exact smul _ _ h


open Submodule in
lemma adjoin_eq_span (s : Set A) : (adjoin R s).toSubmodule = span R (Subsemigroup.closure s) := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    s : Set A
    ⊢ Eq (NonUnitalAlgebra.adjoin R s).toSubmodule (Submodule.span R ↑(Subsemigrou …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      s : Set A
      ⊢ LE.le (NonUnitalAlgebra.adjoin R s).toSubmodule (Submodule.span R ↑(Subsemig …
    -/
  · intro x hx
    induction hx using adjoin_induction with
    | mem x hx => exact subset_span <| Subsemigroup.subset_closure hx
    | add x y _ _ hpx hpy => exact add_mem hpx hpy
    | zero => exact zero_mem _
    | mul x y _ _ hpx hpy =>
      apply span_induction₂ ?Hs (by simp) (by simp) ?Hadd_l ?Hadd_r ?Hsmul_l ?Hsmul_r hpx hpy
      case Hs => exact fun x y hx hy ↦ subset_span <| mul_mem hx hy
      case Hadd_l => exact fun x y z _ _ _ hxz hyz ↦ by simpa [add_mul] using add_mem hxz hyz
      case Hadd_r => exact fun x y z _ _ _ hxz hyz ↦ by simpa [mul_add] using add_mem hxz hyz
      case Hsmul_l => exact fun r x y _ _ hxy ↦ by simpa [smul_mul_assoc] using smul_mem _ _ hxy
      case Hsmul_r => exact fun r x y _ _ hxy ↦ by simpa [mul_smul_comm] using smul_mem _ _ hxy
    | smul r x _ hpx => exact smul_mem _ _ hpx
    /-
      case a
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      s : Set A
      ⊢ LE.le (Submodule.span R ↑(Subsemigroup.closure s)) (NonUnitalAlgebra.adjoin  …
    -/
  · apply span_le.2 _
    /-
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      s : Set A
      ⊢ HasSubset.Subset ↑(Subsemigroup.closure s) ↑(NonUnitalAlgebra.adjoin R s).to …
    -/
    show Subsemigroup.closure s ≤ (adjoin R s).toSubsemigroup
    /-
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      s : Set A
      ⊢ LE.le (Subsemigroup.closure s) (NonUnitalAlgebra.adjoin R s).toSubsemigroup
    -/
    exact Subsemigroup.closure_le.2 (subset_adjoin R)
    /-
      🎉 no goals
    -/


@[simp]
theorem adjoin_empty : adjoin R (∅ : Set A) = ⊥ :=
                         /-
                           R : Type u
                           A : Type v
                           inst✝⁴ : CommSemiring R
                           inst✝³ : NonUnitalNonAssocSemiring A
                           inst✝² : Module R A
                           inst✝¹ : IsScalarTower R A A
                           inst✝ : SMulCommClass R A A
                           ⊢ Eq (NonUnitalAlgebra.adjoin R Bot.bot) Bot.bot
                         -/
  show adjoin R ⊥ = ⊥ by apply GaloisConnection.l_bot; exact NonUnitalAlgebra.gc
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem adjoin_univ : adjoin R (Set.univ : Set A) = ⊤ :=
  eq_top_iff.2 fun _x hx => subset_adjoin R hx


open NonUnitalSubalgebra in
lemma _root_.NonUnitalAlgHom.map_adjoin [IsScalarTower R B B] [SMulCommClass R B B]
    (f : F) (s : Set A) : map f (adjoin R s) = adjoin R (f '' s) :=
  Set.image_preimage.l_comm_of_u_comm (gc_map_comap f) NonUnitalAlgebra.gi.gc
    NonUnitalAlgebra.gi.gc fun _t => rfl


open NonUnitalSubalgebra in
@[simp]
lemma _root_.NonUnitalAlgHom.map_adjoin_singleton [IsScalarTower R B B] [SMulCommClass R B B]
    (f : F) (x : A) : map f (adjoin R {x}) = adjoin R {f x} := by
  /-
    F : Type u_1
    R : Type u
    A : Type v
    B : Type w
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : NonUnitalNonAssocSemiring A
    inst✝⁸ : Module R A
    inst✝⁷ : NonUnitalNonAssocSemiring B
    inst✝⁶ : Module R B
    inst✝⁵ : FunLike F A B
    inst✝⁴ : NonUnitalAlgHomClass F R A B
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : IsScalarTower R B B
    inst✝ : SMulCommClass R B B
    f : F
    x : A
    ⊢ Eq (NonUnitalSubalgebra.map f (NonUnitalAlgebra.adjoin R (Singleton.singleto …
  -/
  simp [NonUnitalAlgHom.map_adjoin]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_top : (↑(⊤ : NonUnitalSubalgebra R A) : Set A) = Set.univ :=
  rfl


@[simp]
theorem mem_top {x : A} : x ∈ (⊤ : NonUnitalSubalgebra R A) :=
  Set.mem_univ x


@[simp]
theorem top_toSubmodule : (⊤ : NonUnitalSubalgebra R A).toSubmodule = ⊤ :=
  rfl


@[simp]
theorem top_toNonUnitalSubsemiring : (⊤ : NonUnitalSubalgebra R A).toNonUnitalSubsemiring = ⊤ :=
  rfl


@[simp]
theorem top_toSubring {R A : Type*} [CommRing R] [NonUnitalNonAssocRing A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A] :
    (⊤ : NonUnitalSubalgebra R A).toNonUnitalSubring = ⊤ :=
  rfl


@[simp]
theorem toSubmodule_eq_top {S : NonUnitalSubalgebra R A} : S.toSubmodule = ⊤ ↔ S = ⊤ :=
  NonUnitalSubalgebra.toSubmodule'.injective.eq_iff' top_toSubmodule


@[simp]
theorem toNonUnitalSubsemiring_eq_top {S : NonUnitalSubalgebra R A} :
    S.toNonUnitalSubsemiring = ⊤ ↔ S = ⊤ :=
  NonUnitalSubalgebra.toNonUnitalSubsemiring_injective.eq_iff' top_toNonUnitalSubsemiring


@[simp]
theorem to_subring_eq_top {R A : Type*} [CommRing R] [Ring A] [Algebra R A]
    {S : NonUnitalSubalgebra R A} : S.toNonUnitalSubring = ⊤ ↔ S = ⊤ :=
  NonUnitalSubalgebra.toNonUnitalSubring_injective.eq_iff' top_toSubring


theorem mem_sup_left {S T : NonUnitalSubalgebra R A} : ∀ {x : A}, x ∈ S → x ∈ S ⊔ T := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    S T : NonUnitalSubalgebra R A
    ⊢ ∀ {x : A}, Membership.mem S x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    S T : NonUnitalSubalgebra R A
    ⊢ LE.le S (Max.max S T)
  -/
  exact le_sup_left
  /-
    🎉 no goals
  -/


theorem mem_sup_right {S T : NonUnitalSubalgebra R A} : ∀ {x : A}, x ∈ T → x ∈ S ⊔ T := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    S T : NonUnitalSubalgebra R A
    ⊢ ∀ {x : A}, Membership.mem T x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    S T : NonUnitalSubalgebra R A
    ⊢ LE.le T (Max.max S T)
  -/
  exact le_sup_right
  /-
    🎉 no goals
  -/


theorem mul_mem_sup {S T : NonUnitalSubalgebra R A} {x y : A} (hx : x ∈ S) (hy : y ∈ T) :
    x * y ∈ S ⊔ T :=
  mul_mem (mem_sup_left hx) (mem_sup_right hy)


theorem map_sup [IsScalarTower R B B] [SMulCommClass R B B]
    (f : F) (S T : NonUnitalSubalgebra R A) :
    ((S ⊔ T).map f : NonUnitalSubalgebra R B) = S.map f ⊔ T.map f :=
  (NonUnitalSubalgebra.gc_map_comap f).l_sup


theorem map_inf [IsScalarTower R B B] [SMulCommClass R B B]
    (f : F) (hf : Function.Injective f) (S T : NonUnitalSubalgebra R A) :
    ((S ⊓ T).map f : NonUnitalSubalgebra R B) = S.map f ⊓ T.map f :=
  SetLike.coe_injective (Set.image_inter hf)


@[simp, norm_cast]
theorem coe_inf (S T : NonUnitalSubalgebra R A) : (↑(S ⊓ T) : Set A) = (S : Set A) ∩ T :=
  rfl


@[simp]
theorem mem_inf {S T : NonUnitalSubalgebra R A} {x : A} : x ∈ S ⊓ T ↔ x ∈ S ∧ x ∈ T :=
  Iff.rfl


@[simp]
theorem inf_toSubmodule (S T : NonUnitalSubalgebra R A) :
    (S ⊓ T).toSubmodule = S.toSubmodule ⊓ T.toSubmodule :=
  rfl


@[simp]
theorem inf_toNonUnitalSubsemiring (S T : NonUnitalSubalgebra R A) :
    (S ⊓ T).toNonUnitalSubsemiring = S.toNonUnitalSubsemiring ⊓ T.toNonUnitalSubsemiring :=
  rfl


@[simp, norm_cast]
theorem coe_sInf (S : Set (NonUnitalSubalgebra R A)) : (↑(sInf S) : Set A) = ⋂ s ∈ S, ↑s :=
  sInf_image


theorem mem_sInf {S : Set (NonUnitalSubalgebra R A)} {x : A} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    S : Set (NonUnitalSubalgebra R A)
    x : A
    ⊢ Iff (Membership.mem (InfSet.sInf S) x) (∀ (p : NonUnitalSubalgebra R A), Mem …
  -/
  simp only [← SetLike.mem_coe, coe_sInf, Set.mem_iInter₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem sInf_toSubmodule (S : Set (NonUnitalSubalgebra R A)) :
    (sInf S).toSubmodule = sInf (NonUnitalSubalgebra.toSubmodule '' S) :=
                              /-
                                R : Type u
                                A : Type v
                                inst✝⁴ : CommSemiring R
                                inst✝³ : NonUnitalNonAssocSemiring A
                                inst✝² : Module R A
                                inst✝¹ : IsScalarTower R A A
                                inst✝ : SMulCommClass R A A
                                S : Set (NonUnitalSubalgebra R A)
                                ⊢ Eq ↑(InfSet.sInf S).toSubmodule ↑(InfSet.sInf (Set.image NonUnitalSubalgebra …
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem sInf_toNonUnitalSubsemiring (S : Set (NonUnitalSubalgebra R A)) :
    (sInf S).toNonUnitalSubsemiring = sInf (NonUnitalSubalgebra.toNonUnitalSubsemiring '' S) :=
                              /-
                                R : Type u
                                A : Type v
                                inst✝⁴ : CommSemiring R
                                inst✝³ : NonUnitalNonAssocSemiring A
                                inst✝² : Module R A
                                inst✝¹ : IsScalarTower R A A
                                inst✝ : SMulCommClass R A A
                                S : Set (NonUnitalSubalgebra R A)
                                ⊢ Eq ↑(InfSet.sInf S).toNonUnitalSubsemiring ↑(InfSet.sInf (Set.image NonUnita …
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → NonUnitalSubalgebra R A} :
                                           /-
                                             R : Type u
                                             A : Type v
                                             inst✝⁴ : CommSemiring R
                                             inst✝³ : NonUnitalNonAssocSemiring A
                                             inst✝² : Module R A
                                             inst✝¹ : IsScalarTower R A A
                                             inst✝ : SMulCommClass R A A
                                             ι : Sort u_2
                                             S : ι → NonUnitalSubalgebra R A
                                             ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
                                           -/
    (↑(⨅ i, S i) : Set A) = ⋂ i, S i := by simp [iInf]
                                           /-
                                             🎉 no goals
                                           -/


theorem mem_iInf {ι : Sort*} {S : ι → NonUnitalSubalgebra R A} {x : A} :
                                        /-
                                          R : Type u
                                          A : Type v
                                          inst✝⁴ : CommSemiring R
                                          inst✝³ : NonUnitalNonAssocSemiring A
                                          inst✝² : Module R A
                                          inst✝¹ : IsScalarTower R A A
                                          inst✝ : SMulCommClass R A A
                                          ι : Sort u_2
                                          S : ι → NonUnitalSubalgebra R A
                                          x : A
                                          ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
                                        -/
    (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by simp only [iInf, mem_sInf, Set.forall_mem_range]
                                        /-
                                          🎉 no goals
                                        -/


theorem map_iInf {ι : Sort*} [Nonempty ι]
    [IsScalarTower R B B] [SMulCommClass R B B] (f : F)
    (hf : Function.Injective f) (S : ι → NonUnitalSubalgebra R A) :
    ((⨅ i, S i).map f : NonUnitalSubalgebra R B) = ⨅ i, (S i).map f := by
  /-
    F : Type u_1
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalNonAssocSemiring A
    inst✝⁹ : Module R A
    inst✝⁸ : NonUnitalNonAssocSemiring B
    inst✝⁷ : Module R B
    inst✝⁶ : FunLike F A B
    inst✝⁵ : NonUnitalAlgHomClass F R A B
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : SMulCommClass R A A
    ι : Sort u_2
    inst✝² : Nonempty ι
    inst✝¹ : IsScalarTower R B B
    inst✝ : SMulCommClass R B B
    f : F
    hf : Function.Injective ⇑f
    S : ι → NonUnitalSubalgebra R A
    ⊢ Eq (NonUnitalSubalgebra.map f (iInf fun i => S i)) (iInf fun i => NonUnitalS …
  -/
  apply SetLike.coe_injective
  /-
    case a
    F : Type u_1
    R : Type u
    A : Type v
    B : Type w
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalNonAssocSemiring A
    inst✝⁹ : Module R A
    inst✝⁸ : NonUnitalNonAssocSemiring B
    inst✝⁷ : Module R B
    inst✝⁶ : FunLike F A B
    inst✝⁵ : NonUnitalAlgHomClass F R A B
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : SMulCommClass R A A
    ι : Sort u_2
    inst✝² : Nonempty ι
    inst✝¹ : IsScalarTower R B B
    inst✝ : SMulCommClass R B B
    f : F
    hf : Function.Injective ⇑f
    S : ι → NonUnitalSubalgebra R A
    ⊢ Eq ↑(NonUnitalSubalgebra.map f (iInf fun i => S i)) ↑(iInf fun i => NonUnita …
  -/
  simpa using (Set.injOn_of_injective hf).image_iInter_eq (s := SetLike.coe ∘ S)
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_toSubmodule {ι : Sort*} (S : ι → NonUnitalSubalgebra R A) :
    (⨅ i, S i).toSubmodule = ⨅ i, (S i).toSubmodule :=
                              /-
                                R : Type u
                                A : Type v
                                inst✝⁴ : CommSemiring R
                                inst✝³ : NonUnitalNonAssocSemiring A
                                inst✝² : Module R A
                                inst✝¹ : IsScalarTower R A A
                                inst✝ : SMulCommClass R A A
                                ι : Sort u_2
                                S : ι → NonUnitalSubalgebra R A
                                ⊢ Eq ↑(iInf fun i => S i).toSubmodule ↑(iInf fun i => (S i).toSubmodule)
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


instance : Inhabited (NonUnitalSubalgebra R A) :=
  ⟨⊥⟩


theorem mem_bot {x : A} : x ∈ (⊥ : NonUnitalSubalgebra R A) ↔ x = 0 :=
  show x ∈ Submodule.span R (NonUnitalSubsemiring.closure (∅ : Set A) : Set A) ↔ x = 0 by
    rw [NonUnitalSubsemiring.closure_empty, NonUnitalSubsemiring.coe_bot,
      Submodule.span_zero_singleton, Submodule.mem_bot]


theorem toSubmodule_bot : (⊥ : NonUnitalSubalgebra R A).toSubmodule = ⊥ := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    ⊢ Eq Bot.bot.toSubmodule Bot.bot
  -/
  ext
  /-
    case h
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    x✝ : A
    ⊢ Iff (Membership.mem Bot.bot.toSubmodule x✝) (Membership.mem Bot.bot x✝)
  -/
  simp only [mem_bot, NonUnitalSubalgebra.mem_toSubmodule, Submodule.mem_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_bot : ((⊥ : NonUnitalSubalgebra R A) : Set A) = {0} := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalNonAssocSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    ⊢ Eq (↑Bot.bot) (Singleton.singleton 0)
  -/
  simp [Set.ext_iff, NonUnitalAlgebra.mem_bot]
  /-
    🎉 no goals
  -/


theorem eq_top_iff {S : NonUnitalSubalgebra R A} : S = ⊤ ↔ ∀ x : A, x ∈ S :=
                 /-
                   R : Type u
                   A : Type v
                   inst✝⁴ : CommSemiring R
                   inst✝³ : NonUnitalNonAssocSemiring A
                   inst✝² : Module R A
                   inst✝¹ : IsScalarTower R A A
                   inst✝ : SMulCommClass R A A
                   S : NonUnitalSubalgebra R A
                   h : Eq S Top.top
                   x : A
                   ⊢ Membership.mem S x
                 -/
                         /-
                           🎉 no goals
                         -/
  ⟨fun h x => by rw [h]; exact mem_top, fun h => by ext x; exact ⟨fun _ => mem_top, fun _ => h x⟩⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem range_id : NonUnitalAlgHom.range (NonUnitalAlgHom.id R A) = ⊤ :=
  SetLike.coe_injective Set.range_id


@[simp]
theorem map_top (f : A →ₙₐ[R] B) : (⊤ : NonUnitalSubalgebra R A).map f = NonUnitalAlgHom.range f :=
  SetLike.coe_injective Set.image_univ


@[simp]
theorem map_bot [IsScalarTower R B B] [SMulCommClass R B B]
    (f : A →ₙₐ[R] B) : (⊥ : NonUnitalSubalgebra R A).map f = ⊥ :=
                              /-
                                R : Type u
                                A : Type v
                                B : Type w
                                inst✝⁸ : CommSemiring R
                                inst✝⁷ : NonUnitalNonAssocSemiring A
                                inst✝⁶ : Module R A
                                inst✝⁵ : NonUnitalNonAssocSemiring B
                                inst✝⁴ : Module R B
                                inst✝³ : IsScalarTower R A A
                                inst✝² : SMulCommClass R A A
                                inst✝¹ : IsScalarTower R B B
                                inst✝ : SMulCommClass R B B
                                f : NonUnitalAlgHom (MonoidHom.id R) A B
                                ⊢ Eq ↑(NonUnitalSubalgebra.map f Bot.bot) ↑Bot.bot
                              -/
  SetLike.coe_injective <| by simp [NonUnitalAlgebra.coe_bot, NonUnitalSubalgebra.coe_map]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem comap_top [IsScalarTower R B B] [SMulCommClass R B B]
    (f : A →ₙₐ[R] B) : (⊤ : NonUnitalSubalgebra R B).comap f = ⊤ :=
  eq_top_iff.2 fun _ => mem_top


/-- `NonUnitalAlgHom` to `⊤ : NonUnitalSubalgebra R A`. -/
def toTop : A →ₙₐ[R] (⊤ : NonUnitalSubalgebra R A) :=
  NonUnitalAlgHom.codRestrict (NonUnitalAlgHom.id R A) ⊤ fun _ => mem_top


theorem range_eq_top [IsScalarTower R B B] [SMulCommClass R B B] (f : A →ₙₐ[R] B) :
    NonUnitalAlgHom.range f = (⊤ : NonUnitalSubalgebra R B) ↔ Function.Surjective f :=
  NonUnitalAlgebra.eq_top_iff


@[deprecated (since := "2024-11-11")] alias range_top_iff_surjective := range_eq_top


theorem range_val : NonUnitalAlgHom.range (NonUnitalSubalgebraClass.subtype S) = S :=
  ext <| Set.ext_iff.1 <| (NonUnitalSubalgebraClass.subtype S).coe_range.trans Subtype.range_val


instance subsingleton_of_subsingleton [Subsingleton A] : Subsingleton (NonUnitalSubalgebra R A) :=
                              /-
                                R : Type u
                                A : Type v
                                B✝ : Type w
                                inst✝³ : CommSemiring R
                                inst✝² : NonUnitalNonAssocSemiring A
                                inst✝¹ : Module R A
                                S : NonUnitalSubalgebra R A
                                inst✝ : Subsingleton A
                                B C : NonUnitalSubalgebra R A
                                x : A
                                ⊢ Iff (Membership.mem B x) (Membership.mem C x)
                              -/
  ⟨fun B C => ext fun x => by simp only [Subsingleton.elim x 0, zero_mem B, zero_mem C]⟩
                              /-
                                🎉 no goals
                              -/


/-- The product of two non-unital subalgebras is a non-unital subalgebra. -/
def prod : NonUnitalSubalgebra R (A × B) :=
  { S.toNonUnitalSubsemiring.prod S₁.toNonUnitalSubsemiring with
    carrier := S ×ˢ S₁
    smul_mem' := fun r _x hx => ⟨SMulMemClass.smul_mem r hx.1, SMulMemClass.smul_mem r hx.2⟩ }


@[simp]
theorem coe_prod : (prod S S₁ : Set (A × B)) = (S : Set A) ×ˢ S₁ :=
  rfl


theorem prod_toSubmodule : (S.prod S₁).toSubmodule = S.toSubmodule.prod S₁.toSubmodule :=
  rfl


@[simp]
theorem mem_prod {S : NonUnitalSubalgebra R A} {S₁ : NonUnitalSubalgebra R B} {x : A × B} :
    x ∈ prod S S₁ ↔ x.1 ∈ S ∧ x.2 ∈ S₁ :=
  Set.mem_prod


@[simp]
                                                                        /-
                                                                          R : Type u
                                                                          A : Type v
                                                                          B : Type w
                                                                          inst✝⁸ : CommSemiring R
                                                                          inst✝⁷ : NonUnitalNonAssocSemiring A
                                                                          inst✝⁶ : Module R A
                                                                          inst✝⁵ : NonUnitalNonAssocSemiring B
                                                                          inst✝⁴ : Module R B
                                                                          inst✝³ : IsScalarTower R A A
                                                                          inst✝² : SMulCommClass R A A
                                                                          inst✝¹ : IsScalarTower R B B
                                                                          inst✝ : SMulCommClass R B B
                                                                          ⊢ Eq (Top.top.prod Top.top) Top.top
                                                                        -/
theorem prod_top : (prod ⊤ ⊤ : NonUnitalSubalgebra R (A × B)) = ⊤ := by ext; simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem prod_mono {S T : NonUnitalSubalgebra R A} {S₁ T₁ : NonUnitalSubalgebra R B} :
    S ≤ T → S₁ ≤ T₁ → prod S S₁ ≤ prod T T₁ :=
  Set.prod_mono


@[simp]
theorem prod_inf_prod {S T : NonUnitalSubalgebra R A} {S₁ T₁ : NonUnitalSubalgebra R B} :
    S.prod S₁ ⊓ T.prod T₁ = (S ⊓ T).prod (S₁ ⊓ T₁) :=
  SetLike.coe_injective Set.prod_inter_prod


instance _root_.NonUnitalAlgHom.subsingleton [Subsingleton (NonUnitalSubalgebra R A)] :
    Subsingleton (A →ₙₐ[R] B) :=
  ⟨fun f g =>
    NonUnitalAlgHom.ext fun a =>
      have : a ∈ (⊥ : NonUnitalSubalgebra R A) :=
        Subsingleton.elim (⊤ : NonUnitalSubalgebra R A) ⊥ ▸ mem_top
      (mem_bot.mp this).symm ▸ (map_zero f).trans (map_zero g).symm⟩



/-- The map `S → T` when `S` is a non-unital subalgebra contained in the non-unital subalgebra `T`.

This is the non-unital subalgebra version of `Submodule.inclusion`, or `Subring.inclusion`  -/
def inclusion {S T : NonUnitalSubalgebra R A} (h : S ≤ T) : S →ₙₐ[R] T where
  toFun := Set.inclusion h
  map_add' _ _ := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_smul' _ _ := rfl


theorem inclusion_injective {S T : NonUnitalSubalgebra R A} (h : S ≤ T) :
    Function.Injective (inclusion h) := fun _ _ => Subtype.ext ∘ Subtype.mk.inj


@[simp]
theorem inclusion_self {S : NonUnitalSubalgebra R A} :
    inclusion (le_refl S) = NonUnitalAlgHom.id R S :=
  rfl


@[simp]
theorem inclusion_mk {S T : NonUnitalSubalgebra R A} (h : S ≤ T) (x : A) (hx : x ∈ S) :
    inclusion h ⟨x, hx⟩ = ⟨x, h hx⟩ :=
  rfl


theorem inclusion_right {S T : NonUnitalSubalgebra R A} (h : S ≤ T) (x : T) (m : (x : A) ∈ S) :
    inclusion h ⟨x, m⟩ = x :=
  Subtype.ext rfl


@[simp]
theorem inclusion_inclusion {S T U : NonUnitalSubalgebra R A} (hst : S ≤ T) (htu : T ≤ U) (x : S) :
    inclusion htu (inclusion hst x) = inclusion (le_trans hst htu) x :=
  Subtype.ext rfl


@[simp]
theorem coe_inclusion {S T : NonUnitalSubalgebra R A} (h : S ≤ T) (s : S) :
    (inclusion h s : A) = s :=
  rfl


theorem coe_iSup_of_directed [Nonempty ι] {S : ι → NonUnitalSubalgebra R A}
    (dir : Directed (· ≤ ·) S) : ↑(iSup S) = ⋃ i, (S i : Set A) :=
  let K : NonUnitalSubalgebra R A :=
    { __ := NonUnitalSubsemiring.copy _ _ (NonUnitalSubsemiring.coe_iSup_of_directed dir).symm
      smul_mem' := fun r _x hx ↦
        let ⟨i, hi⟩ := Set.mem_iUnion.1 hx
        Set.mem_iUnion.2 ⟨i, (S i).smul_mem' r hi⟩ }
  have : iSup S = K := le_antisymm
    (iSup_le fun i ↦ le_iSup (fun i ↦ (S i : Set A)) i) (Set.iUnion_subset fun _ ↦ le_iSup S _)
  this.symm ▸ rfl


/-- Define an algebra homomorphism on a directed supremum of non-unital subalgebras by defining
it on each non-unital subalgebra, and proving that it agrees on the intersection of
non-unital subalgebras. -/
noncomputable def iSupLift [Nonempty ι] (K : ι → NonUnitalSubalgebra R A) (dir : Directed (· ≤ ·) K)
    (f : ∀ i, K i →ₙₐ[R] B) (hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h))
    (T : NonUnitalSubalgebra R A) (hT : T = iSup K) : ↥T →ₙₐ[R] B := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    S : NonUnitalSubalgebra R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    ⊢ NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.mem T x) B
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
          dsimp
          exact Set.iUnionLift_const _ (fun i : ι => (0 : K i)) (fun _ => rfl) _ (by simp)
        map_mul' := by
          dsimp
          apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· * ·))
          all_goals simp
        map_add' := by
          dsimp
          apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· + ·))
          all_goals simp
        map_smul' := fun r => by
          dsimp
          apply Set.iUnionLift_unary (coe_iSup_of_directed dir) _ (fun _ x => r • x)
            (fun _ _ => rfl)
          all_goals simp }


@[simp]
theorem iSupLift_inclusion {i : ι} (x : K i) (h : K i ≤ T) :
    iSupLift K dir f hf T hT (inclusion h x) = f i x := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) T
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf T hT) ((NonUnitalSubalgebra.inc …
  -/
  subst T
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) (iSup K)
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf (iSup K) ⋯) ((NonUnitalSubalgeb …
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
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
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
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
    (iSupLift K dir f hf T hT).comp (inclusion h) = f i := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    h : LE.le (K i) T
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf T hT).comp (NonUnitalSubalgebra …
  -/
  ext
  /-
    case h
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    h : LE.le (K i) T
    x✝ : Subtype fun x => Membership.mem (K i) x
    ⊢ Eq (((NonUnitalSubalgebra.iSupLift K dir f hf T hT).comp (NonUnitalSubalgebr …
  -/
  simp only [NonUnitalAlgHom.comp_apply, iSupLift_inclusion]
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
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem T ↑x
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf T hT) ⟨↑x, hx⟩) ((f i) x)
  -/
  subst hT
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem (iSup K) ↑x
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf (iSup K) ⋯) ⟨↑x, hx⟩) ((f i) x)
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
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
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    T : NonUnitalSubalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem T x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf T hT) x) ((f i) ⟨↑x, hx⟩)
  -/
  subst hT
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    i : ι
    x : Subtype fun x => Membership.mem (iSup K) x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq ((NonUnitalSubalgebra.iSupLift K dir f hf (iSup K) ⋯) x) ((f i) ⟨↑x, hx⟩)
  -/
  dsimp [iSupLift]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NonUnitalNonAssocSemiring A
    inst✝⁵ : Module R A
    inst✝⁴ : NonUnitalNonAssocSemiring B
    inst✝³ : Module R B
    inst✝² : IsScalarTower R A A
    inst✝¹ : SMulCommClass R A A
    ι : Type u_1
    inst✝ : Nonempty ι
    K : ι → NonUnitalSubalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → NonUnitalAlgHom (MonoidHom.id R) (Subtype fun x => Membership.me …
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (NonUnitalSubal …
    i : ι
    x : Subtype fun x => Membership.mem (iSup K) x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑(iSup K) ⋯ x) ( …
  -/
  apply Set.iUnionLift_of_mem
  /-
    🎉 no goals
  -/


theorem _root_.Set.smul_mem_center (r : R) {a : A} (ha : a ∈ Set.center A) :
    r • a ∈ Set.center A where
               /-
                 R : Type u_1
                 A : Type u_2
                 inst✝⁴ : CommSemiring R
                 inst✝³ : NonUnitalNonAssocSemiring A
                 inst✝² : Module R A
                 inst✝¹ : IsScalarTower R A A
                 inst✝ : SMulCommClass R A A
                 r : R
                 a : A
                 ha : Membership.mem (Set.center A) a
                 b : A
                 ⊢ Eq (HMul.hMul (HSMul.hSMul r a) b) (HMul.hMul b (HSMul.hSMul r a))
               -/
  comm b := by rw [mul_smul_comm, smul_mul_assoc, ha.comm]
               /-
                 🎉 no goals
               -/
                       /-
                         R : Type u_1
                         A : Type u_2
                         inst✝⁴ : CommSemiring R
                         inst✝³ : NonUnitalNonAssocSemiring A
                         inst✝² : Module R A
                         inst✝¹ : IsScalarTower R A A
                         inst✝ : SMulCommClass R A A
                         r : R
                         a : A
                         ha : Membership.mem (Set.center A) a
                         b c : A
                         ⊢ Eq (HMul.hMul (HSMul.hSMul r a) (HMul.hMul b c)) (HMul.hMul (HMul.hMul (HSMu …
                       -/
  left_assoc b c := by rw [smul_mul_assoc, smul_mul_assoc, smul_mul_assoc, ha.left_assoc]
                       /-
                         🎉 no goals
                       -/
  mid_assoc b c := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      r : R
      a : A
      ha : Membership.mem (Set.center A) a
      b c : A
      ⊢ Eq (HMul.hMul (HMul.hMul b (HSMul.hSMul r a)) c) (HMul.hMul b (HMul.hMul (HS …
    -/
    rw [mul_smul_comm, smul_mul_assoc, smul_mul_assoc, mul_smul_comm, ha.mid_assoc]
    /-
      🎉 no goals
    -/
  right_assoc b c := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      r : R
      a : A
      ha : Membership.mem (Set.center A) a
      b c : A
      ⊢ Eq (HMul.hMul (HMul.hMul b c) (HSMul.hSMul r a)) (HMul.hMul b (HMul.hMul c ( …
    -/
    rw [mul_smul_comm, mul_smul_comm, mul_smul_comm, ha.right_assoc]
    /-
      🎉 no goals
    -/


variable (R A) in
/-- The center of a non-unital algebra is the set of elements which commute with every element.
They form a non-unital subalgebra. -/
def center : NonUnitalSubalgebra R A :=
  { NonUnitalSubsemiring.center A with smul_mem' := Set.smul_mem_center }


theorem coe_center : (center R A : Set A) = Set.center A :=
  rfl


/-- The center of a non-unital algebra is commutative and associative -/
instance center.instNonUnitalCommSemiring : NonUnitalCommSemiring (center R A) :=
  NonUnitalSubsemiring.center.instNonUnitalCommSemiring _


instance center.instNonUnitalCommRing {A : Type*} [NonUnitalNonAssocRing A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A] : NonUnitalCommRing (center R A) :=
  NonUnitalSubring.center.instNonUnitalCommRing _


@[simp]
theorem center_toNonUnitalSubsemiring :
    (center R A).toNonUnitalSubsemiring = NonUnitalSubsemiring.center A :=
  rfl


@[simp] lemma center_toNonUnitalSubring (R A : Type*) [CommRing R] [NonUnitalRing A]
    [Module R A] [IsScalarTower R A A] [SMulCommClass R A A] :
    (center R A).toNonUnitalSubring = NonUnitalSubring.center A :=
  rfl


@[simp]
theorem center_eq_top (A : Type*) [NonUnitalCommSemiring A] [Module R A] [IsScalarTower R A A]
    [SMulCommClass R A A] : center R A = ⊤ :=
  SetLike.coe_injective (Set.center_eq_univ A)


theorem mem_center_iff {a : A} : a ∈ center R A ↔ ∀ b : A, b * a = a * b :=
  Subsemigroup.mem_center_iff


@[simp]
theorem _root_.Set.smul_mem_centralizer {s : Set A} (r : R) {a : A} (ha : a ∈ s.centralizer) :
    r • a ∈ s.centralizer :=
                 /-
                   R : Type u_1
                   A : Type u_2
                   inst✝⁴ : CommSemiring R
                   inst✝³ : NonUnitalSemiring A
                   inst✝² : Module R A
                   inst✝¹ : IsScalarTower R A A
                   inst✝ : SMulCommClass R A A
                   s : Set A
                   r : R
                   a : A
                   ha : Membership.mem s.centralizer a
                   x : A
                   hx : Membership.mem s x
                   ⊢ Eq (HMul.hMul x (HSMul.hSMul r a)) (HMul.hMul (HSMul.hSMul r a) x)
                 -/
  fun x hx => by rw [mul_smul_comm, smul_mul_assoc, ha x hx]
                 /-
                   🎉 no goals
                 -/


/-- The centralizer of a set as a non-unital subalgebra. -/
def centralizer (s : Set A) : NonUnitalSubalgebra R A where
  toNonUnitalSubsemiring := NonUnitalSubsemiring.centralizer s
  smul_mem' := Set.smul_mem_centralizer


@[simp, norm_cast]
theorem coe_centralizer (s : Set A) : (centralizer R s : Set A) = s.centralizer :=
  rfl


theorem mem_centralizer_iff {s : Set A} {z : A} : z ∈ centralizer R s ↔ ∀ g ∈ s, g * z = z * g :=
  Iff.rfl


theorem centralizer_le (s t : Set A) (h : s ⊆ t) : centralizer R t ≤ centralizer R s :=
  Set.centralizer_subset h


@[simp]
theorem centralizer_univ : centralizer R Set.univ = center R A :=
  SetLike.ext' (Set.centralizer_univ A)


variable (R) in
lemma adjoin_le_centralizer_centralizer (s : Set A) :
    adjoin R s ≤ centralizer R (centralizer R s) :=
  adjoin_le Set.subset_centralizer_centralizer


lemma commute_of_mem_adjoin_of_forall_mem_commute {a b : A} {s : Set A}
    (hb : b ∈ adjoin R s) (h : ∀ b ∈ s, Commute a b) :
    Commute a b := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a b : A
    s : Set A
    hb : Membership.mem (NonUnitalAlgebra.adjoin R s) b
    h : ∀ (b : A), Membership.mem s b → Commute a b
    ⊢ Commute a b
  -/
  have : a ∈ centralizer R s := by simpa only [Commute.symm_iff (a := a)] using h
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a b : A
    s : Set A
    hb : Membership.mem (NonUnitalAlgebra.adjoin R s) b
    h : ∀ (b : A), Membership.mem s b → Commute a b
    this : Membership.mem (NonUnitalSubalgebra.centralizer R s) a
    ⊢ Commute a b
  -/
  exact adjoin_le_centralizer_centralizer R s hb a this
  /-
    🎉 no goals
  -/


lemma commute_of_mem_adjoin_singleton_of_commute {a b c : A}
    (hc : c ∈ adjoin R {b}) (h : Commute a b) :
    Commute a c :=
                                                       /-
                                                         R : Type u_1
                                                         A : Type u_2
                                                         inst✝⁴ : CommSemiring R
                                                         inst✝³ : NonUnitalSemiring A
                                                         inst✝² : Module R A
                                                         inst✝¹ : IsScalarTower R A A
                                                         inst✝ : SMulCommClass R A A
                                                         a b c : A
                                                         hc : Membership.mem (NonUnitalAlgebra.adjoin R (Singleton.singleton b)) c
                                                         h : Commute a b
                                                         ⊢ ∀ (b_1 : A), Membership.mem (Singleton.singleton b) b_1 → Commute a b_1
                                                       -/
  commute_of_mem_adjoin_of_forall_mem_commute hc <| by simpa
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma commute_of_mem_adjoin_self {a b : A} (hb : b ∈ adjoin R {a}) :
    Commute a b :=
  commute_of_mem_adjoin_singleton_of_commute hb rfl


variable (R) in

/-- If all elements of `s : Set A` commute pairwise, then `adjoin R s` is a non-unital commutative
semiring.

See note [reducible non-instances]. -/
abbrev adjoinNonUnitalCommSemiringOfComm {s : Set A} (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a) :
    NonUnitalCommSemiring (adjoin R s) :=
  { (adjoin R s).toNonUnitalSemiring with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦
      have := adjoin_le_centralizer_centralizer R s
      Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ (this h₁) _ (this h₂) }


/-- If all elements of `s : Set A` commute pairwise, then `adjoin R s` is a non-unital commutative
ring.

See note [reducible non-instances]. -/
abbrev adjoinNonUnitalCommRingOfComm (R : Type*) {A : Type*} [CommRing R] [NonUnitalRing A]
    [Module R A] [IsScalarTower R A A] [SMulCommClass R A A] {s : Set A}
    (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a) : NonUnitalCommRing (adjoin R s) :=
  { (adjoin R s).toNonUnitalRing, adjoinNonUnitalCommSemiringOfComm R hcomm with }


/-- A non-unital subsemiring is a non-unital `ℕ`-subalgebra. -/
def nonUnitalSubalgebraOfNonUnitalSubsemiring (S : NonUnitalSubsemiring R) :
    NonUnitalSubalgebra ℕ R where
  toNonUnitalSubsemiring := S
  smul_mem' n _x hx := nsmul_mem (S := S) hx n


@[simp]
theorem mem_nonUnitalSubalgebraOfNonUnitalSubsemiring {x : R} {S : NonUnitalSubsemiring R} :
    x ∈ nonUnitalSubalgebraOfNonUnitalSubsemiring S ↔ x ∈ S :=
  Iff.rfl


/-- A non-unital subring is a non-unital `ℤ`-subalgebra. -/
def nonUnitalSubalgebraOfNonUnitalSubring (S : NonUnitalSubring R) : NonUnitalSubalgebra ℤ R where
  toNonUnitalSubsemiring := S.toNonUnitalSubsemiring
  smul_mem' n _x hx := zsmul_mem (K := S) hx n


@[simp]
theorem mem_nonUnitalSubalgebraOfNonUnitalSubring {x : R} {S : NonUnitalSubring R} :
    x ∈ nonUnitalSubalgebraOfNonUnitalSubring S ↔ x ∈ S :=
  Iff.rfl


