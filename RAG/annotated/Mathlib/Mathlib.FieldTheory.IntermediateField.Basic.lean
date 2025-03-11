/-- `S : IntermediateField K L` is a subset of `L` such that there is a field
tower `L / S / K`. -/
structure IntermediateField extends Subalgebra K L where
  inv_mem' : ∀ x ∈ carrier, x⁻¹ ∈ carrier


instance : SetLike (IntermediateField K L) L :=
  ⟨fun S => S.toSubalgebra.carrier, by
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S : IntermediateField K L
      ⊢ Function.Injective fun S => S.carrier
    -/
    rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩
    /-
      case mk.mk.mk.mk
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S : IntermediateField K L
      toSubsemiring✝¹ : Subsemiring L
      algebraMap_mem'✝¹ : ∀ (r : K), Membership.mem toSubsemiring✝¹.carrier ((algebr …
      inv_mem'✝¹ : ∀ (x : L), Membership.mem { toSubsemiring := toSubsemiring✝¹, alg …
      toSubsemiring✝ : Subsemiring L
      algebraMap_mem'✝ : ∀ (r : K), Membership.mem toSubsemiring✝.carrier ((algebraM …
      inv_mem'✝ : ∀ (x : L), Membership.mem { toSubsemiring := toSubsemiring✝, algeb …
      ⊢ Eq ((fun S => S.carrier) { toSubsemiring := toSubsemiring✝¹, algebraMap_mem' …
    -/
    simp ⟩
    /-
      🎉 no goals
    -/


protected theorem neg_mem {x : L} (hx : x ∈ S) : -x ∈ S := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : IntermediateField K L
    x : L
    hx : Membership.mem S x
    ⊢ Membership.mem S (Neg.neg x)
  -/
  show -x ∈S.toSubalgebra; simpa
                           /-
                             🎉 no goals
                           -/


/-- Reinterpret an `IntermediateField` as a `Subfield`. -/
def toSubfield : Subfield L :=
  { S.toSubalgebra with
    neg_mem' := S.neg_mem,
    inv_mem' := S.inv_mem' }


instance : SubfieldClass (IntermediateField K L) L where
  add_mem {s} := s.add_mem'
  zero_mem {s} := s.zero_mem'
  neg_mem {s} := s.neg_mem
  mul_mem {s} := s.mul_mem'
  one_mem {s} := s.one_mem'
  inv_mem {s} := s.inv_mem' _

--@[simp] Porting note (https://github.com/leanprover-community/mathlib4/issues/10618): simp can prove it

theorem mem_carrier {s : IntermediateField K L} {x : L} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


/-- Two intermediate fields are equal if they have the same elements. -/
@[ext]
theorem ext {S T : IntermediateField K L} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
theorem coe_toSubalgebra : (S.toSubalgebra : Set L) = S :=
  rfl


@[simp]
theorem coe_toSubfield : (S.toSubfield : Set L) = S :=
  rfl


@[simp]
theorem mem_mk (s : Subsemiring L) (hK : ∀ x, algebraMap K L x ∈ s) (hi) (x : L) :
    x ∈ IntermediateField.mk (Subalgebra.mk s hK) hi ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem mem_toSubalgebra (s : IntermediateField K L) (x : L) : x ∈ s.toSubalgebra ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem mem_toSubfield (s : IntermediateField K L) (x : L) : x ∈ s.toSubfield ↔ x ∈ s :=
  Iff.rfl


/-- Copy of an intermediate field with a new `carrier` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (S : IntermediateField K L) (s : Set L) (hs : s = ↑S) :
    IntermediateField K L where
  toSubalgebra := S.toSubalgebra.copy s hs
  inv_mem' := hs.symm ▸ S.inv_mem'


@[simp]
theorem coe_copy (S : IntermediateField K L) (s : Set L) (hs : s = ↑S) :
    (S.copy s hs : Set L) = s :=
  rfl


theorem copy_eq (S : IntermediateField K L) (s : Set L) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


/-- An intermediate field contains the image of the smaller field. -/
theorem algebraMap_mem (x : K) : algebraMap K L x ∈ S :=
  S.algebraMap_mem' x


/-- An intermediate field is closed under scalar multiplication. -/
theorem smul_mem {y : L} : y ∈ S → ∀ {x : K}, x • y ∈ S :=
  S.toSubalgebra.smul_mem


/-- An intermediate field contains the ring's 1. -/
protected theorem one_mem : (1 : L) ∈ S :=
  one_mem S


/-- An intermediate field contains the ring's 0. -/
protected theorem zero_mem : (0 : L) ∈ S :=
  zero_mem S


/-- An intermediate field is closed under multiplication. -/
protected theorem mul_mem {x y : L} : x ∈ S → y ∈ S → x * y ∈ S :=
  mul_mem


/-- An intermediate field is closed under addition. -/
protected theorem add_mem {x y : L} : x ∈ S → y ∈ S → x + y ∈ S :=
  add_mem


/-- An intermediate field is closed under subtraction -/
protected theorem sub_mem {x y : L} : x ∈ S → y ∈ S → x - y ∈ S :=
  sub_mem


/-- An intermediate field is closed under inverses. -/
protected theorem inv_mem {x : L} : x ∈ S → x⁻¹ ∈ S :=
  inv_mem


/-- An intermediate field is closed under division. -/
protected theorem div_mem {x y : L} : x ∈ S → y ∈ S → x / y ∈ S :=
  div_mem


/-- Product of a list of elements in an intermediate_field is in the intermediate_field. -/
protected theorem list_prod_mem {l : List L} : (∀ x ∈ l, x ∈ S) → l.prod ∈ S :=
  list_prod_mem


/-- Sum of a list of elements in an intermediate field is in the intermediate_field. -/
protected theorem list_sum_mem {l : List L} : (∀ x ∈ l, x ∈ S) → l.sum ∈ S :=
  list_sum_mem


/-- Product of a multiset of elements in an intermediate field is in the intermediate_field. -/
protected theorem multiset_prod_mem (m : Multiset L) : (∀ a ∈ m, a ∈ S) → m.prod ∈ S :=
  multiset_prod_mem m


/-- Sum of a multiset of elements in an `IntermediateField` is in the `IntermediateField`. -/
protected theorem multiset_sum_mem (m : Multiset L) : (∀ a ∈ m, a ∈ S) → m.sum ∈ S :=
  multiset_sum_mem m


/-- Product of elements of an intermediate field indexed by a `Finset` is in the intermediate_field.
-/
protected theorem prod_mem {ι : Type*} {t : Finset ι} {f : ι → L} (h : ∀ c ∈ t, f c ∈ S) :
    (∏ i ∈ t, f i) ∈ S :=
  prod_mem h


/-- Sum of elements in an `IntermediateField` indexed by a `Finset` is in the `IntermediateField`.
-/
protected theorem sum_mem {ι : Type*} {t : Finset ι} {f : ι → L} (h : ∀ c ∈ t, f c ∈ S) :
    (∑ i ∈ t, f i) ∈ S :=
  sum_mem h


protected theorem pow_mem {x : L} (hx : x ∈ S) (n : ℤ) : x ^ n ∈ S :=
  zpow_mem hx n


protected theorem zsmul_mem {x : L} (hx : x ∈ S) (n : ℤ) : n • x ∈ S :=
  zsmul_mem hx n


protected theorem intCast_mem (n : ℤ) : (n : L) ∈ S :=
  intCast_mem S n


protected theorem coe_add (x y : S) : (↑(x + y) : L) = ↑x + ↑y :=
  rfl


protected theorem coe_neg (x : S) : (↑(-x) : L) = -↑x :=
  rfl


protected theorem coe_mul (x y : S) : (↑(x * y) : L) = ↑x * ↑y :=
  rfl


protected theorem coe_inv (x : S) : (↑x⁻¹ : L) = (↑x)⁻¹ :=
  rfl


protected theorem coe_zero : ((0 : S) : L) = 0 :=
  rfl


protected theorem coe_one : ((1 : S) : L) = 1 :=
  rfl


protected theorem coe_pow (x : S) (n : ℕ) : (↑(x ^ n : S) : L) = (x : L) ^ n :=
  SubmonoidClass.coe_pow x n


                                                /-
                                                  K : Type u_1
                                                  L : Type u_2
                                                  inst✝² : Field K
                                                  inst✝¹ : Field L
                                                  inst✝ : Algebra K L
                                                  S : IntermediateField K L
                                                  n : Nat
                                                  ⊢ Membership.mem S ↑n
                                                -/
theorem natCast_mem (n : ℕ) : (n : L) ∈ S := by simpa using intCast_mem S n
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated _root_.natCast_mem (since := "2024-04-05")] alias coe_nat_mem := natCast_mem

@[deprecated _root_.intCast_mem (since := "2024-04-05")] alias coe_int_mem := intCast_mem


/-- Turn a subalgebra closed under inverses into an intermediate field -/
def Subalgebra.toIntermediateField (S : Subalgebra K L) (inv_mem : ∀ x ∈ S, x⁻¹ ∈ S) :
    IntermediateField K L :=
  { S with
    inv_mem' := inv_mem }


@[simp]
theorem toSubalgebra_toIntermediateField (S : Subalgebra K L) (inv_mem : ∀ x ∈ S, x⁻¹ ∈ S) :
    (S.toIntermediateField inv_mem).toSubalgebra = S := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Subalgebra K L
    inv_mem : ∀ (x : L), Membership.mem S x → Membership.mem S (Inv.inv x)
    ⊢ Eq (S.toIntermediateField inv_mem).toSubalgebra S
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Subalgebra K L
    inv_mem : ∀ (x : L), Membership.mem S x → Membership.mem S (Inv.inv x)
    x✝ : L
    ⊢ Iff (Membership.mem (S.toIntermediateField inv_mem).toSubalgebra x✝) (Member …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toIntermediateField_toSubalgebra (S : IntermediateField K L) :
    (S.toSubalgebra.toIntermediateField fun _ => S.inv_mem) = S := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : IntermediateField K L
    ⊢ Eq (S.toIntermediateField ⋯) S
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : IntermediateField K L
    x✝ : L
    ⊢ Iff (Membership.mem (S.toIntermediateField ⋯) x✝) (Membership.mem S x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Turn a subalgebra satisfying `IsField` into an intermediate_field -/
def Subalgebra.toIntermediateField' (S : Subalgebra K L) (hS : IsField S) : IntermediateField K L :=
  S.toIntermediateField fun x hx => by
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      S : Subalgebra K L
      hS : IsField (Subtype fun x => Membership.mem S x)
      x : L
      hx : Membership.mem S x
      ⊢ Membership.mem S (Inv.inv x)
    -/
    by_cases hx0 : x = 0
      /-
        case pos
        K : Type u_1
        L : Type u_2
        L' : Type u_3
        inst✝⁴ : Field K
        inst✝³ : Field L
        inst✝² : Field L'
        inst✝¹ : Algebra K L
        inst✝ : Algebra K L'
        S✝ : IntermediateField K L
        S : Subalgebra K L
        hS : IsField (Subtype fun x => Membership.mem S x)
        x : L
        hx : Membership.mem S x
        hx0 : Eq x 0
        ⊢ Membership.mem S (Inv.inv x)
      -/
    · rw [hx0, inv_zero]
      /-
        case pos
        K : Type u_1
        L : Type u_2
        L' : Type u_3
        inst✝⁴ : Field K
        inst✝³ : Field L
        inst✝² : Field L'
        inst✝¹ : Algebra K L
        inst✝ : Algebra K L'
        S✝ : IntermediateField K L
        S : Subalgebra K L
        hS : IsField (Subtype fun x => Membership.mem S x)
        x : L
        hx : Membership.mem S x
        hx0 : Eq x 0
        ⊢ Membership.mem S 0
      -/
      exact S.zero_mem
      /-
        🎉 no goals
      -/
    /-
      case neg
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      S : Subalgebra K L
      hS : IsField (Subtype fun x => Membership.mem S x)
      x : L
      hx : Membership.mem S x
      hx0 : Not (Eq x 0)
      ⊢ Membership.mem S (Inv.inv x)
    -/
    letI hS' := hS.toField
    /-
      case neg
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      S : Subalgebra K L
      hS : IsField (Subtype fun x => Membership.mem S x)
      x : L
      hx : Membership.mem S x
      hx0 : Not (Eq x 0)
      hS' : Field (Subtype fun x => Membership.mem S x) := hS.toField
      ⊢ Membership.mem S (Inv.inv x)
    -/
    obtain ⟨y, hy⟩ := hS.mul_inv_cancel (show (⟨x, hx⟩ : S) ≠ 0 from Subtype.coe_ne_coe.1 hx0)
    /-
      case neg.intro
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      S : Subalgebra K L
      hS : IsField (Subtype fun x => Membership.mem S x)
      x : L
      hx : Membership.mem S x
      hx0 : Not (Eq x 0)
      hS' : Field (Subtype fun x => Membership.mem S x) := hS.toField
      y : Subtype fun x => Membership.mem S x
      hy : Eq (HMul.hMul ⟨x, hx⟩ y) 1
      ⊢ Membership.mem S (Inv.inv x)
    -/
    rw [Subtype.ext_iff, S.coe_mul, S.coe_one, Subtype.coe_mk, mul_eq_one_iff_inv_eq₀ hx0] at hy
    /-
      case neg.intro
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      S : Subalgebra K L
      hS : IsField (Subtype fun x => Membership.mem S x)
      x : L
      hx : Membership.mem S x
      hx0 : Not (Eq x 0)
      hS' : Field (Subtype fun x => Membership.mem S x) := hS.toField
      y : Subtype fun x => Membership.mem S x
      hy : Eq (Inv.inv x) ↑y
      ⊢ Membership.mem S (Inv.inv x)
    -/
    exact hy.symm ▸ y.2
    /-
      🎉 no goals
    -/


@[simp]
theorem toSubalgebra_toIntermediateField' (S : Subalgebra K L) (hS : IsField S) :
    (S.toIntermediateField' hS).toSubalgebra = S := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Subalgebra K L
    hS : IsField (Subtype fun x => Membership.mem S x)
    ⊢ Eq (S.toIntermediateField' hS).toSubalgebra S
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Subalgebra K L
    hS : IsField (Subtype fun x => Membership.mem S x)
    x✝ : L
    ⊢ Iff (Membership.mem (S.toIntermediateField' hS).toSubalgebra x✝) (Membership …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toIntermediateField'_toSubalgebra (S : IntermediateField K L) :
    S.toSubalgebra.toIntermediateField' (Field.toIsField S) = S := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : IntermediateField K L
    ⊢ Eq (S.toIntermediateField' ⋯) S
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : IntermediateField K L
    x✝ : L
    ⊢ Iff (Membership.mem (S.toIntermediateField' ⋯) x✝) (Membership.mem S x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Turn a subfield of `L` containing the image of `K` into an intermediate field -/
def Subfield.toIntermediateField (S : Subfield L) (algebra_map_mem : ∀ x, algebraMap K L x ∈ S) :
    IntermediateField K L :=
  { S with
    algebraMap_mem' := algebra_map_mem }


/-- An intermediate field inherits a field structure -/
instance toField : Field S :=
  S.toSubfield.toField


@[norm_cast]
theorem coe_sum {ι : Type*} [Fintype ι] (f : ι → S) : (↑(∑ i, f i) : L) = ∑ i, (f i : L) := by
  classical
    induction' (Finset.univ : Finset ι) using Finset.induction_on with i s hi H
    · simp
    · rw [Finset.sum_insert hi, AddMemClass.coe_add, H, Finset.sum_insert hi]


@[norm_cast]
theorem coe_prod {ι : Type*} [Fintype ι] (f : ι → S) : (↑(∏ i, f i) : L) = ∏ i, (f i : L) := by
  classical
    induction' (Finset.univ : Finset ι) using Finset.induction_on with i s hi H
    · simp
    · rw [Finset.prod_insert hi, MulMemClass.coe_mul, H, Finset.prod_insert hi]


/-- The action by an intermediate field is the action by the underlying field. -/
instance [SMul L X] (F : IntermediateField K L) : SMul F X :=
  inferInstanceAs (SMul F.toSubfield X)


theorem smul_def [SMul L X] {F : IntermediateField K L} (g : F) (m : X) : g • m = (g : L) • m :=
  rfl


instance smulCommClass_left [SMul L Y] [SMul X Y] [SMulCommClass L X Y]
    (F : IntermediateField K L) : SMulCommClass F X Y :=
  inferInstanceAs (SMulCommClass F.toSubfield X Y)


instance smulCommClass_right [SMul X Y] [SMul L Y] [SMulCommClass X L Y]
    (F : IntermediateField K L) : SMulCommClass X F Y :=
  inferInstanceAs (SMulCommClass X F.toSubfield Y)

-- note: giving this instance the default priority may trigger trouble with synthesizing instances
-- for field extensions with more than one intermediate field. For example, in a field extension
-- `E/F`, and with `K₁ ≤ K₂` of type `IntermediateField F E`, this instance will cause a search
-- for `IsScalarTower K₁ K₂ E` to trigger a search for `IsScalarTower E K₂ E` which may
-- take a long time to fail.

/-- Note that this provides `IsScalarTower F K K` which is needed by `smul_mul_assoc`. -/
instance (priority := 900) [SMul X Y] [SMul L X] [SMul L Y] [IsScalarTower L X Y]
    (F : IntermediateField K L) : IsScalarTower F X Y :=
  inferInstanceAs (IsScalarTower F.toSubfield X Y)


instance [SMul L X] [FaithfulSMul L X] (F : IntermediateField K L) : FaithfulSMul F X :=
  inferInstanceAs (FaithfulSMul F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [MulAction L X] (F : IntermediateField K L) : MulAction F X :=
  inferInstanceAs (MulAction F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [AddMonoid X] [DistribMulAction L X] (F : IntermediateField K L) : DistribMulAction F X :=
  inferInstanceAs (DistribMulAction F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [Monoid X] [MulDistribMulAction L X] (F : IntermediateField K L) :
    MulDistribMulAction F X :=
  inferInstanceAs (MulDistribMulAction F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [Zero X] [SMulWithZero L X] (F : IntermediateField K L) : SMulWithZero F X :=
  inferInstanceAs (SMulWithZero F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [Zero X] [MulActionWithZero L X] (F : IntermediateField K L) : MulActionWithZero F X :=
  inferInstanceAs (MulActionWithZero F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [AddCommMonoid X] [Module L X] (F : IntermediateField K L) : Module F X :=
  inferInstanceAs (Module F.toSubfield X)


/-- The action by an intermediate field is the action by the underlying field. -/
instance [Semiring X] [MulSemiringAction L X] (F : IntermediateField K L) : MulSemiringAction F X :=
  inferInstanceAs (MulSemiringAction F.toSubfield X)


instance toAlgebra : Algebra S L :=
  inferInstanceAs (Algebra S.toSubalgebra L)


instance module' {R} [Semiring R] [SMul R K] [Module R L] [IsScalarTower R K L] : Module R S :=
  inferInstanceAs (Module R S.toSubalgebra)


instance algebra' {R' K L : Type*} [Field K] [Field L] [Algebra K L] (S : IntermediateField K L)
    [CommSemiring R'] [SMul R' K] [Algebra R' L] [IsScalarTower R' K L] : Algebra R' S :=
  inferInstanceAs (Algebra R' S.toSubalgebra)


instance isScalarTower {R} [Semiring R] [SMul R K] [Module R L] [IsScalarTower R K L] :
    IsScalarTower R K S :=
  inferInstanceAs (IsScalarTower R K S.toSubalgebra)


@[simp]
theorem coe_smul {R} [Semiring R] [SMul R K] [Module R L] [IsScalarTower R K L] (r : R) (x : S) :
    ↑(r • x : S) = (r • (x : L)) :=
  rfl


@[simp] lemma algebraMap_apply (x : S) : algebraMap S L x = x := rfl


@[simp] lemma coe_algebraMap_apply (x : K) : ↑(algebraMap K S x) = algebraMap K L x := rfl


instance isScalarTower_bot {R : Type*} [Semiring R] [Algebra L R] : IsScalarTower S L R :=
  IsScalarTower.subalgebra _ _ _ S.toSubalgebra


instance isScalarTower_mid {R : Type*} [Semiring R] [Algebra L R] [Algebra K R]
    [IsScalarTower K L R] : IsScalarTower K S R :=
  IsScalarTower.subalgebra' _ _ _ S.toSubalgebra


/-- Specialize `is_scalar_tower_mid` to the common case where the top field is `L` -/
instance isScalarTower_mid' : IsScalarTower K S L :=
  inferInstance


instance {E} [Semiring E] [Algebra L E] : Algebra S E := inferInstanceAs (Algebra S.toSubalgebra E)


instance : Algebra S T := T.algebra

instance : Module S T := Algebra.toModule

instance : SMul S T := Algebra.toSMul

instance [Algebra K E] [IsScalarTower K L E] : IsScalarTower K S T := T.isScalarTower

/-- Given `f : L →ₐ[K] L'`, `S.comap f` is the intermediate field between `K` and `L`
  such that `f x ∈ S ↔ x ∈ S.comap f`. -/
def comap (f : L →ₐ[K] L') (S : IntermediateField K L') : IntermediateField K L where
  __ := S.toSubalgebra.comap f
                                     /-
                                       K : Type u_1
                                       L : Type u_2
                                       L' : Type u_3
                                       inst✝⁴ : Field K
                                       inst✝³ : Field L
                                       inst✝² : Field L'
                                       inst✝¹ : Algebra K L
                                       inst✝ : Algebra K L'
                                       S✝ : IntermediateField K L
                                       X : ?m.121796
                                       Y : ?m.121799
                                       f : AlgHom K L L'
                                       S : IntermediateField K L'
                                       x : L
                                       hx : Membership.mem __spread✝⁻⁰.carrier x
                                       ⊢ Membership.mem S (f (Inv.inv x))
                                     -/
  inv_mem' x hx := show f x⁻¹ ∈ S by rw [map_inv₀ f x]; exact S.inv_mem hx
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Given `f : L →ₐ[K] L'`, `S.map f` is the intermediate field between `K` and `L'`
such that `x ∈ S ↔ f x ∈ S.map f`. -/
def map (f : L →ₐ[K] L') (S : IntermediateField K L) : IntermediateField K L' where
  __ := S.toSubalgebra.map f
  inv_mem' := by
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      X : ?m.123794
      Y : ?m.123797
      f : AlgHom K L L'
      S : IntermediateField K L
      ⊢ ∀ (x : L'), Membership.mem __spread✝⁻⁰.carrier x → Membership.mem __spread✝⁻ …
    -/
    rintro _ ⟨x, hx, rfl⟩
    /-
      case intro.intro
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S✝ : IntermediateField K L
      X : ?m.123794
      Y : ?m.123797
      f : AlgHom K L L'
      S : IntermediateField K L
      x : L
      hx : Membership.mem (↑S.toSubsemiring) x
      ⊢ Membership.mem __spread✝⁻⁰.carrier (Inv.inv (↑f x))
    -/
    exact ⟨x⁻¹, S.inv_mem hx, map_inv₀ f x⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_map (f : L →ₐ[K] L') : (S.map f : Set L') = f '' S :=
  rfl


@[simp]
theorem toSubalgebra_map (f : L →ₐ[K] L') : (S.map f).toSubalgebra = S.toSubalgebra.map f :=
  rfl


@[simp]
theorem toSubfield_map (f : L →ₐ[K] L') : (S.map f).toSubfield = S.toSubfield.map f :=
  rfl


theorem map_map {K L₁ L₂ L₃ : Type*} [Field K] [Field L₁] [Algebra K L₁] [Field L₂] [Algebra K L₂]
    [Field L₃] [Algebra K L₃] (E : IntermediateField K L₁) (f : L₁ →ₐ[K] L₂) (g : L₂ →ₐ[K] L₃) :
    (E.map f).map g = E.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image _ _ _


theorem map_mono (f : L →ₐ[K] L') {S T : IntermediateField K L} (h : S ≤ T) :
    S.map f ≤ T.map f :=
  SetLike.coe_mono (Set.image_subset f h)


theorem map_le_iff_le_comap {f : L →ₐ[K] L'}
    {s : IntermediateField K L} {t : IntermediateField K L'} :
    s.map f ≤ t ↔ s ≤ t.comap f :=
  Set.image_subset_iff


theorem gc_map_comap (f : L →ₐ[K] L') : GaloisConnection (map f) (comap f) :=
  fun _ _ ↦ map_le_iff_le_comap


/-- Given an equivalence `e : L ≃ₐ[K] L'` of `K`-field extensions and an intermediate
field `E` of `L/K`, `intermediateFieldMap e E` is the induced equivalence
between `E` and `E.map e` -/
def intermediateFieldMap (e : L ≃ₐ[K] L') (E : IntermediateField K L) : E ≃ₐ[K] E.map e.toAlgHom :=
  e.subalgebraMap E.toSubalgebra

/- We manually add these two simp lemmas because `@[simps]` before `intermediate_field_map`
  led to a timeout. -/
-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem intermediateFieldMap_apply_coe (e : L ≃ₐ[K] L') (E : IntermediateField K L) (a : E) :
    ↑(intermediateFieldMap e E a) = e a :=
  rfl

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem intermediateFieldMap_symm_apply_coe (e : L ≃ₐ[K] L') (E : IntermediateField K L)
    (a : E.map e.toAlgHom) : ↑((intermediateFieldMap e E).symm a) = e.symm a :=
  rfl


/-- The range of an algebra homomorphism, as an intermediate field. -/
@[simps toSubalgebra]
def fieldRange : IntermediateField K L' :=
  { f.range, (f : L →+* L').fieldRange with }


@[simp]
theorem coe_fieldRange : ↑f.fieldRange = Set.range f :=
  rfl


@[simp]
theorem fieldRange_toSubfield : f.fieldRange.toSubfield = (f : L →+* L').fieldRange :=
  rfl


@[simp]
theorem mem_fieldRange {y : L'} : y ∈ f.fieldRange ↔ ∃ x, f x = y :=
  Iff.rfl


/-- The embedding from an intermediate field of `L / K` to `L`. -/
def val : S →ₐ[K] L :=
  S.toSubalgebra.val


@[simp]
theorem coe_val : ⇑S.val = ((↑) : S → L) :=
  rfl


@[simp]
theorem val_mk {x : L} (hx : x ∈ S) : S.val ⟨x, hx⟩ = x :=
  rfl


theorem range_val : S.val.range = S.toSubalgebra :=
  S.toSubalgebra.range_val


@[simp]
theorem fieldRange_val : S.val.fieldRange = S :=
  SetLike.ext' Subtype.range_val


instance AlgHom.inhabited : Inhabited (S →ₐ[K] L) :=
  ⟨S.val⟩


theorem aeval_coe {R : Type*} [CommRing R] [Algebra R K] [Algebra R L] [IsScalarTower R K L]
    (x : S) (P : R[X]) : aeval (x : L) P = aeval x P :=
  aeval_algHom_apply (S.val.restrictScalars R) x P


/-- The map `E → F` when `E` is an intermediate field contained in the intermediate field `F`.

This is the intermediate field version of `Subalgebra.inclusion`. -/
def inclusion {E F : IntermediateField K L} (hEF : E ≤ F) : E →ₐ[K] F :=
  Subalgebra.inclusion hEF


theorem inclusion_injective {E F : IntermediateField K L} (hEF : E ≤ F) :
    Function.Injective (inclusion hEF) :=
  Subalgebra.inclusion_injective hEF


@[simp]
theorem inclusion_self {E : IntermediateField K L} : inclusion (le_refl E) = AlgHom.id K E :=
  Subalgebra.inclusion_self


@[simp]
theorem inclusion_inclusion {E F G : IntermediateField K L} (hEF : E ≤ F) (hFG : F ≤ G) (x : E) :
    inclusion hFG (inclusion hEF x) = inclusion (le_trans hEF hFG) x :=
  Subalgebra.inclusion_inclusion hEF hFG x


@[simp]
theorem coe_inclusion {E F : IntermediateField K L} (hEF : E ≤ F) (e : E) :
    (inclusion hEF e : L) = e :=
  rfl


theorem toSubalgebra_injective : Function.Injective (toSubalgebra : IntermediateField K L → _) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Function.Injective IntermediateField.toSubalgebra
  -/
  intro _ _ h
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    a₁✝ a₂✝ : IntermediateField K L
    h : Eq a₁✝.toSubalgebra a₂✝.toSubalgebra
    ⊢ Eq a₁✝ a₂✝
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    a₁✝ a₂✝ : IntermediateField K L
    h : Eq a₁✝.toSubalgebra a₂✝.toSubalgebra
    x✝ : L
    ⊢ Iff (Membership.mem a₁✝ x✝) (Membership.mem a₂✝ x✝)
  -/
  simp_rw [← mem_toSubalgebra, h]
  /-
    🎉 no goals
  -/


theorem toSubfield_injective : Function.Injective (toSubfield : IntermediateField K L → _) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Function.Injective IntermediateField.toSubfield
  -/
  intro _ _ h
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    a₁✝ a₂✝ : IntermediateField K L
    h : Eq a₁✝.toSubfield a₂✝.toSubfield
    ⊢ Eq a₁✝ a₂✝
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    a₁✝ a₂✝ : IntermediateField K L
    h : Eq a₁✝.toSubfield a₂✝.toSubfield
    x✝ : L
    ⊢ Iff (Membership.mem a₁✝ x✝) (Membership.mem a₂✝ x✝)
  -/
  simp_rw [← mem_toSubfield, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem toSubalgebra_inj : F.toSubalgebra = E.toSubalgebra ↔ F = E := toSubalgebra_injective.eq_iff


@[deprecated (since := "2024-12-29")] alias toSubalgebra_eq_iff := toSubalgebra_inj


@[simp]
theorem toSubfield_inj : F.toSubfield = E.toSubfield ↔ F = E := toSubfield_injective.eq_iff


theorem map_injective (f : L →ₐ[K] L') : Function.Injective (map f) := by
  /-
    K : Type u_1
    L : Type u_2
    L' : Type u_3
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Field L'
    inst✝¹ : Algebra K L
    inst✝ : Algebra K L'
    f : AlgHom K L L'
    ⊢ Function.Injective (IntermediateField.map f)
  -/
  intro _ _ h
  rwa [← toSubalgebra_injective.eq_iff, toSubalgebra_map, toSubalgebra_map,
    (Subalgebra.map_injective f.injective).eq_iff, toSubalgebra_inj] at h


theorem set_range_subset : Set.range (algebraMap K L) ⊆ S :=
  S.toSubalgebra.range_subset


theorem fieldRange_le : (algebraMap K L).fieldRange ≤ S.toSubfield := fun x hx =>
                                  /-
                                    K : Type u_1
                                    L : Type u_2
                                    inst✝² : Field K
                                    inst✝¹ : Field L
                                    inst✝ : Algebra K L
                                    S : IntermediateField K L
                                    x : L
                                    hx : Membership.mem (algebraMap K L).fieldRange x
                                    ⊢ Membership.mem (Set.range ⇑(algebraMap K L)) x
                                  -/
  S.toSubalgebra.range_subset (by rwa [Set.mem_range, ← RingHom.mem_fieldRange])
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem toSubalgebra_le_toSubalgebra {S S' : IntermediateField K L} :
    S.toSubalgebra ≤ S'.toSubalgebra ↔ S ≤ S' :=
  Iff.rfl


@[simp]
theorem toSubalgebra_lt_toSubalgebra {S S' : IntermediateField K L} :
    S.toSubalgebra < S'.toSubalgebra ↔ S < S' :=
  Iff.rfl


/-- Lift an intermediate_field of an intermediate_field -/
def lift {F : IntermediateField K L} (E : IntermediateField K F) : IntermediateField K L :=
  E.map (val F)

-- Porting note: change from `HasLiftT` to `CoeOut`

instance hasLift {F : IntermediateField K L} :
    CoeOut (IntermediateField K F) (IntermediateField K L) :=
  ⟨lift⟩


theorem lift_injective (F : IntermediateField K L) : Function.Injective F.lift :=
  map_injective F.val


theorem lift_le {F : IntermediateField K L} (E : IntermediateField K F) : lift E ≤ F := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    F : IntermediateField K L
    E : IntermediateField K (Subtype fun x => Membership.mem F x)
    ⊢ LE.le (IntermediateField.lift E) F
  -/
  rintro _ ⟨x, _, rfl⟩
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    F : IntermediateField K L
    E : IntermediateField K (Subtype fun x => Membership.mem F x)
    x : Subtype fun x => Membership.mem F x
    left✝ : Membership.mem (↑E.toSubsemiring) x
    ⊢ Membership.mem F (↑F.val x)
  -/
  exact x.2
  /-
    🎉 no goals
  -/


theorem mem_lift {F : IntermediateField K L} {E : IntermediateField K F} (x : F) :
    x.1 ∈ lift E ↔ x ∈ E :=
  Subtype.val_injective.mem_set_image


/--The algEquiv between an intermediate field and its lift-/
def liftAlgEquiv {E : IntermediateField K L} (F : IntermediateField K E) : ↥F ≃ₐ[K] lift F where
  toFun x := ⟨x.1.1, (mem_lift x.1).mpr x.2⟩
  invFun x := ⟨⟨x.1, lift_le F x.2⟩, (mem_lift ⟨x.1, lift_le F x.2⟩).mp x.2⟩
  left_inv := congrFun rfl
  right_inv := congrFun rfl
  map_mul' _ _ := rfl
  map_add' _ _ := rfl
  commutes' _ := rfl


lemma liftAlgEquiv_apply {E : IntermediateField K L} (F : IntermediateField K E) (x : F) :
    (liftAlgEquiv F x).1 = x := rfl


/-- Given a tower `L / ↥E / L' / K` of field extensions, where `E` is an `L'`-intermediate field of
`L`, reinterpret `E` as a `K`-intermediate field of `L`. -/
def restrictScalars (E : IntermediateField L' L) : IntermediateField K L :=
  { E.toSubfield, E.toSubalgebra.restrictScalars K with
    carrier := E.carrier }


@[simp]
theorem coe_restrictScalars {E : IntermediateField L' L} :
    (restrictScalars K E : Set L) = (E : Set L) :=
  rfl


@[simp]
theorem restrictScalars_toSubalgebra {E : IntermediateField L' L} :
    (E.restrictScalars K).toSubalgebra = E.toSubalgebra.restrictScalars K :=
  SetLike.coe_injective rfl


@[simp]
theorem restrictScalars_toSubfield {E : IntermediateField L' L} :
    (E.restrictScalars K).toSubfield = E.toSubfield :=
  SetLike.coe_injective rfl


@[simp]
theorem mem_restrictScalars {E : IntermediateField L' L} {x : L} :
    x ∈ restrictScalars K E ↔ x ∈ E :=
  Iff.rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars K : IntermediateField L' L → IntermediateField K L) :=
                               /-
                                 K : Type u_1
                                 L : Type u_2
                                 L' : Type u_3
                                 inst✝⁶ : Field K
                                 inst✝⁵ : Field L
                                 inst✝⁴ : Field L'
                                 inst✝³ : Algebra K L
                                 inst✝² : Algebra K L'
                                 inst✝¹ : Algebra L' L
                                 inst✝ : IsScalarTower K L' L
                                 U V : IntermediateField L' L
                                 H : Eq (IntermediateField.restrictScalars K U) (IntermediateField.restrictScal …
                                 x : L
                                 ⊢ Iff (Membership.mem U x) (Membership.mem V x)
                               -/
  fun U V H => ext fun x => by rw [← mem_restrictScalars K, H, mem_restrictScalars]
                               /-
                                 🎉 no goals
                               -/


/-- If `F ≤ E` are two subfields of `L`, then `E` is also an intermediate field of
`L / F`. It can be viewed as an inverse to `IntermediateField.toSubfield`. -/
def extendScalars : IntermediateField F L := E.toIntermediateField fun ⟨_, hf⟩ ↦ h hf


@[simp]
theorem coe_extendScalars : (extendScalars h : Set L) = (E : Set L) := rfl


@[simp]
theorem extendScalars_toSubfield : (extendScalars h).toSubfield = E := SetLike.coe_injective rfl


@[simp]
theorem mem_extendScalars : x ∈ extendScalars h ↔ x ∈ E := Iff.rfl


theorem extendScalars_le_extendScalars_iff : extendScalars h ≤ extendScalars h' ↔ E ≤ E' := Iff.rfl


theorem extendScalars_le_iff (E' : IntermediateField F L) :
    extendScalars h ≤ E' ↔ E ≤ E'.toSubfield := Iff.rfl


theorem le_extendScalars_iff (E' : IntermediateField F L) :
    E' ≤ extendScalars h ↔ E'.toSubfield ≤ E := Iff.rfl


/-- `Subfield.extendScalars.orderIso` bundles `Subfield.extendScalars`
into an order isomorphism from
`{ E : Subfield L // F ≤ E }` to `IntermediateField F L`. Its inverse is
`IntermediateField.toSubfield`. -/
@[simps]
def extendScalars.orderIso :
    { E : Subfield L // F ≤ E } ≃o IntermediateField F L where
  toFun E := extendScalars E.2
  invFun E := ⟨E.toSubfield, fun x hx ↦ E.algebraMap_mem ⟨x, hx⟩⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' {E E'} := by
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S : IntermediateField K L
      F E✝ E'✝ : Subfield L
      h : LE.le F E✝
      h' : LE.le F E'✝
      x : L
      E E' : Subtype fun E => LE.le F E
      ⊢ Iff (LE.le ({ toFun := fun E => Subfield.extendScalars ⋯, invFun := fun E => …
    -/
    simp only [Equiv.coe_fn_mk]
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S : IntermediateField K L
      F E✝ E'✝ : Subfield L
      h : LE.le F E✝
      h' : LE.le F E'✝
      x : L
      E E' : Subtype fun E => LE.le F E
      ⊢ Iff (LE.le (Subfield.extendScalars ⋯) (Subfield.extendScalars ⋯)) (LE.le E E')
    -/
    exact extendScalars_le_extendScalars_iff _ _
    /-
      🎉 no goals
    -/


theorem extendScalars_injective :
    Function.Injective fun E : { E : Subfield L // F ≤ E } ↦ extendScalars E.2 :=
  (extendScalars.orderIso F).injective


/-- If `F ≤ E` are two intermediate fields of `L / K`, then `E` is also an intermediate field of
`L / F`. It can be viewed as an inverse to `IntermediateField.restrictScalars`. -/
def extendScalars : IntermediateField F L :=
  Subfield.extendScalars (show F.toSubfield ≤ E.toSubfield from h)


@[simp]
theorem extendScalars_toSubfield : (extendScalars h).toSubfield = E.toSubfield :=
  SetLike.coe_injective rfl


@[simp]
theorem extendScalars_restrictScalars : (extendScalars h).restrictScalars K = E := rfl


theorem extendScalars_le_iff (E' : IntermediateField F L) :
    extendScalars h ≤ E' ↔ E ≤ E'.restrictScalars K := Iff.rfl


theorem le_extendScalars_iff (E' : IntermediateField F L) :
    E' ≤ extendScalars h ↔ E'.restrictScalars K ≤ E := Iff.rfl


/-- `IntermediateField.extendScalars.orderIso` bundles `IntermediateField.extendScalars`
into an order isomorphism from
`{ E : IntermediateField K L // F ≤ E }` to `IntermediateField F L`. Its inverse is
`IntermediateField.restrictScalars`. -/
@[simps]
def extendScalars.orderIso : { E : IntermediateField K L // F ≤ E } ≃o IntermediateField F L where
  toFun E := extendScalars E.2
  invFun E := ⟨E.restrictScalars K, fun x hx ↦ E.algebraMap_mem ⟨x, hx⟩⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' {E E'} := by
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S F E✝ E'✝ : IntermediateField K L
      h : LE.le F E✝
      h' : LE.le F E'✝
      x : L
      E E' : Subtype fun E => LE.le F E
      ⊢ Iff (LE.le ({ toFun := fun E => IntermediateField.extendScalars ⋯, invFun := …
    -/
    simp only [Equiv.coe_fn_mk]
    /-
      K : Type u_1
      L : Type u_2
      L' : Type u_3
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field L'
      inst✝¹ : Algebra K L
      inst✝ : Algebra K L'
      S F E✝ E'✝ : IntermediateField K L
      h : LE.le F E✝
      h' : LE.le F E'✝
      x : L
      E E' : Subtype fun E => LE.le F E
      ⊢ Iff (LE.le (IntermediateField.extendScalars ⋯) (IntermediateField.extendScal …
    -/
    exact extendScalars_le_extendScalars_iff _ _
    /-
      🎉 no goals
    -/


theorem extendScalars_injective :
    Function.Injective fun E : { E : IntermediateField K L // F ≤ E } ↦ extendScalars E.2 :=
  (extendScalars.orderIso F).injective


/--
If `F ≤ E` are two intermediate fields of `L / K`, then `F` is also an intermediate field of
`E / K`. It is an inverse of `IntermediateField.lift`, and can be viewed as a dual to
`IntermediateField.extendScalars`.
-/
def restrict : IntermediateField K E :=
  (IntermediateField.inclusion h).fieldRange


theorem mem_restrict (x : E) : x ∈ restrict h ↔ x.1 ∈ F :=
  Set.ext_iff.mp (Set.range_inclusion h) x


@[simp]
theorem lift_restrict : lift (restrict h) = F := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    F E : IntermediateField K L
    h : LE.le F E
    ⊢ Eq (IntermediateField.lift (IntermediateField.restrict h)) F
  -/
  ext x
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    F E : IntermediateField K L
    h : LE.le F E
    x : L
    ⊢ Iff (Membership.mem (IntermediateField.lift (IntermediateField.restrict h))  …
  -/
  refine ⟨fun hx ↦ ?_, fun hx ↦ ?_⟩
    /-
      case h.refine_1
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      F E : IntermediateField K L
      h : LE.le F E
      x : L
      hx : Membership.mem (IntermediateField.lift (IntermediateField.restrict h)) x
      ⊢ Membership.mem F x
    -/
  · let y : E := ⟨x, lift_le (restrict h) hx⟩
    /-
      case h.refine_1
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      F E : IntermediateField K L
      h : LE.le F E
      x : L
      hx : Membership.mem (IntermediateField.lift (IntermediateField.restrict h)) x
      y : Subtype fun x => Membership.mem E x := ⟨x, ⋯⟩
      ⊢ Membership.mem F x
    -/
    exact (mem_restrict h y).1 ((mem_lift y).1 hx)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      F E : IntermediateField K L
      h : LE.le F E
      x : L
      hx : Membership.mem F x
      ⊢ Membership.mem (IntermediateField.lift (IntermediateField.restrict h)) x
    -/
  · let y : E := ⟨x, h hx⟩
    /-
      case h.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      F E : IntermediateField K L
      h : LE.le F E
      x : L
      hx : Membership.mem F x
      y : Subtype fun x => Membership.mem E x := ⟨x, ⋯⟩
      ⊢ Membership.mem (IntermediateField.lift (IntermediateField.restrict h)) x
    -/
    exact (mem_lift y).2 ((mem_restrict h y).2 hx)
    /-
      🎉 no goals
    -/


/--
`F` is equivalent to `F` as an intermediate field of `E / K`.
-/
noncomputable def restrict_algEquiv :
    F ≃ₐ[K] ↥(IntermediateField.restrict h) :=
  AlgEquiv.ofInjectiveField _


