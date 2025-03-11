/-- `SubfieldClass S K` states `S` is a type of subsets `s ⊆ K` closed under field operations. -/
class SubfieldClass (S K : Type*) [DivisionRing K] [SetLike S K] extends SubringClass S K,
  InvMemClass S K : Prop


/-- A subfield contains `1`, products and inverses.

Be assured that we're not actually proving that subfields are subgroups:
`SubgroupClass` is really an abbreviation of `SubgroupWithOrWithoutZeroClass`.
 -/
instance (priority := 100) toSubgroupClass : SubgroupClass S K :=
  { h with }


@[aesop safe apply (rule_sets := [SetLike])]
lemma nnratCast_mem (s : S) (q : ℚ≥0) : (q : K) ∈ s := by
  /-
    K : Type u
    inst✝¹ : DivisionRing K
    S : Type u_1
    inst✝ : SetLike S K
    h : SubfieldClass S K
    s : S
    q : NNRat
    ⊢ Membership.mem s ↑q
  -/
  simpa only [NNRat.cast_def] using div_mem (natCast_mem s q.num) (natCast_mem s q.den)
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [SetLike])]
lemma ratCast_mem (s : S) (q : ℚ) : (q : K) ∈ s := by
  /-
    K : Type u
    inst✝¹ : DivisionRing K
    S : Type u_1
    inst✝ : SetLike S K
    h : SubfieldClass S K
    s : S
    q : Rat
    ⊢ Membership.mem s ↑q
  -/
  simpa only [Rat.cast_def] using div_mem (intCast_mem s q.num) (natCast_mem s q.den)
  /-
    🎉 no goals
  -/


instance instNNRatCast (s : S) : NNRatCast s where nnratCast q := ⟨q, nnratCast_mem s q⟩

instance instRatCast (s : S) : RatCast s where ratCast q := ⟨q, ratCast_mem s q⟩


@[simp, norm_cast] lemma coe_nnratCast (s : S) (q : ℚ≥0) : ((q : s) : K) = q := rfl

@[simp, norm_cast] lemma coe_ratCast (s : S) (x : ℚ) : ((x : s) : K) = x := rfl


@[aesop safe apply (rule_sets := [SetLike])]
lemma nnqsmul_mem (s : S) (q : ℚ≥0) (hx : x ∈ s) : q • x ∈ s := by
  /-
    K : Type u
    inst✝¹ : DivisionRing K
    S : Type u_1
    inst✝ : SetLike S K
    h : SubfieldClass S K
    x : K
    s : S
    q : NNRat
    hx : Membership.mem s x
    ⊢ Membership.mem s (HSMul.hSMul q x)
  -/
  simpa only [NNRat.smul_def] using mul_mem (nnratCast_mem _ _) hx
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [SetLike])]
lemma qsmul_mem (s : S) (q : ℚ) (hx : x ∈ s) : q • x ∈ s := by
  /-
    K : Type u
    inst✝¹ : DivisionRing K
    S : Type u_1
    inst✝ : SetLike S K
    h : SubfieldClass S K
    x : K
    s : S
    q : Rat
    hx : Membership.mem s x
    ⊢ Membership.mem s (HSMul.hSMul q x)
  -/
  simpa only [Rat.smul_def] using mul_mem (ratCast_mem _ _) hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-05")] alias coe_rat_cast := coe_ratCast

@[deprecated (since := "2024-04-05")] alias coe_rat_mem := ratCast_mem

@[deprecated (since := "2024-04-05")] alias rat_smul_mem := qsmul_mem


@[aesop safe apply (rule_sets := [SetLike])]
lemma ofScientific_mem (s : S) {b : Bool} {n m : ℕ} :
    (OfScientific.ofScientific n b m : K) ∈ s :=
  SubfieldClass.nnratCast_mem s (OfScientific.ofScientific n b m)


instance instSMulNNRat (s : S) : SMul ℚ≥0 s where smul q x := ⟨q • x, nnqsmul_mem s q x.2⟩

instance instSMulRat (s : S) : SMul ℚ s where smul q x := ⟨q • x, qsmul_mem s q x.2⟩


@[simp, norm_cast] lemma coe_nnqsmul (s : S) (q : ℚ≥0) (x : s) : ↑(q • x) = q • (x : K) := rfl

@[simp, norm_cast] lemma coe_qsmul (s : S) (q : ℚ) (x : s) : ↑(q • x) = q • (x : K) := rfl


/-- A subfield inherits a division ring structure -/
instance (priority := 75) toDivisionRing (s : S) : DivisionRing s :=
  Subtype.coe_injective.divisionRing ((↑) : s → K)
    rfl rfl (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (coe_nnqsmul _) (coe_qsmul _) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl)

-- Prefer subclasses of `Field` over subclasses of `SubfieldClass`.

/-- A subfield of a field inherits a field structure -/
instance (priority := 75) toField {K} [Field K] [SetLike S K] [SubfieldClass S K] (s : S) :
    Field s :=
  Subtype.coe_injective.field ((↑) : s → K)
    rfl rfl (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (coe_nnqsmul _) (coe_qsmul _) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl)


/-- `Subfield R` is the type of subfields of `R`. A subfield of `R` is a subset `s` that is a
  multiplicative submonoid and an additive subgroup. Note in particular that it shares the
  same 0 and 1 as R. -/
@[stacks 09FD "second part"]
structure Subfield (K : Type u) [DivisionRing K] extends Subring K where
  /-- A subfield is closed under multiplicative inverses. -/
  inv_mem' : ∀ x ∈ carrier, x⁻¹ ∈ carrier


/-- The underlying `AddSubgroup` of a subfield. -/
def toAddSubgroup (s : Subfield K) : AddSubgroup K :=
  { s.toSubring.toAddSubgroup with }

-- Porting note: toSubmonoid already exists
-- /-- The underlying submonoid of a subfield. -/
-- def toSubmonoid (s : Subfield K) : Submonoid K :=
--   { s.toSubring.toSubmonoid with }


instance : SetLike (Subfield K) K where
  coe s := s.carrier
                             /-
                               K : Type u
                               L : Type v
                               M : Type w
                               inst✝² : DivisionRing K
                               inst✝¹ : DivisionRing L
                               inst✝ : DivisionRing M
                               p q : Subfield K
                               h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance : SubfieldClass (Subfield K) K where
  add_mem {s} := s.add_mem'
  zero_mem s := s.zero_mem'
  neg_mem {s} := s.neg_mem'
  mul_mem {s} := s.mul_mem'
  one_mem s := s.one_mem'
  inv_mem {s} := s.inv_mem' _


theorem mem_carrier {s : Subfield K} {x : K} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl

-- Porting note: in lean 3, `S` was type `Set K`

@[simp]
theorem mem_mk {S : Subring K} {x : K} (h) : x ∈ (⟨S, h⟩ : Subfield K) ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_set_mk (S : Subring K) (h) : ((⟨S, h⟩ : Subfield K) : Set K) = S :=
  rfl


@[simp]
theorem mk_le_mk {S S' : Subring K} (h h') : (⟨S, h⟩ : Subfield K) ≤ (⟨S', h'⟩ : Subfield K) ↔
    S ≤ S' :=
  Iff.rfl


/-- Two subfields are equal if they have the same elements. -/
@[ext]
theorem ext {S T : Subfield K} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


/-- Copy of a subfield with a new `carrier` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (S : Subfield K) (s : Set K) (hs : s = ↑S) : Subfield K :=
  { S.toSubring.copy s hs with
    carrier := s
    inv_mem' := hs.symm ▸ S.inv_mem' }


@[simp]
theorem coe_copy (S : Subfield K) (s : Set K) (hs : s = ↑S) : (S.copy s hs : Set K) = s :=
  rfl


theorem copy_eq (S : Subfield K) (s : Set K) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


@[simp]
theorem coe_toSubring (s : Subfield K) : (s.toSubring : Set K) = s :=
  rfl


@[simp]
theorem mem_toSubring (s : Subfield K) (x : K) : x ∈ s.toSubring ↔ x ∈ s :=
  Iff.rfl


/-- A `Subring` containing inverses is a `Subfield`. -/
def Subring.toSubfield (s : Subring K) (hinv : ∀ x ∈ s, x⁻¹ ∈ s) : Subfield K :=
  { s with inv_mem' := hinv }


/-- A subfield contains the field's 1. -/
protected theorem one_mem : (1 : K) ∈ s :=
  one_mem s


/-- A subfield contains the field's 0. -/
protected theorem zero_mem : (0 : K) ∈ s :=
  zero_mem s


/-- A subfield is closed under multiplication. -/
protected theorem mul_mem {x y : K} : x ∈ s → y ∈ s → x * y ∈ s :=
  mul_mem


/-- A subfield is closed under addition. -/
protected theorem add_mem {x y : K} : x ∈ s → y ∈ s → x + y ∈ s :=
  add_mem


/-- A subfield is closed under negation. -/
protected theorem neg_mem {x : K} : x ∈ s → -x ∈ s :=
  neg_mem


/-- A subfield is closed under subtraction. -/
protected theorem sub_mem {x y : K} : x ∈ s → y ∈ s → x - y ∈ s :=
  sub_mem


/-- A subfield is closed under inverses. -/
protected theorem inv_mem {x : K} : x ∈ s → x⁻¹ ∈ s :=
  inv_mem


/-- A subfield is closed under division. -/
protected theorem div_mem {x y : K} : x ∈ s → y ∈ s → x / y ∈ s :=
  div_mem


protected theorem pow_mem {x : K} (hx : x ∈ s) (n : ℕ) : x ^ n ∈ s :=
  pow_mem hx n


protected theorem zsmul_mem {x : K} (hx : x ∈ s) (n : ℤ) : n • x ∈ s :=
  zsmul_mem hx n


protected theorem intCast_mem (n : ℤ) : (n : K) ∈ s := intCast_mem s n


@[deprecated (since := "2024-04-05")] alias coe_int_mem := intCast_mem


theorem zpow_mem {x : K} (hx : x ∈ s) (n : ℤ) : x ^ n ∈ s := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    s : Subfield K
    x : K
    hx : Membership.mem s x
    n : Int
    ⊢ Membership.mem s (HPow.hPow x n)
  -/
  cases n
    /-
      case ofNat
      K : Type u
      inst✝ : DivisionRing K
      s : Subfield K
      x : K
      hx : Membership.mem s x
      a✝ : Nat
      ⊢ Membership.mem s (HPow.hPow x (Int.ofNat a✝))
    -/
  · simpa using s.pow_mem hx _
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      K : Type u
      inst✝ : DivisionRing K
      s : Subfield K
      x : K
      hx : Membership.mem s x
      a✝ : Nat
      ⊢ Membership.mem s (HPow.hPow x (Int.negSucc a✝))
    -/
  · simpa [pow_succ'] using s.inv_mem (s.mul_mem hx (s.pow_mem hx _))
    /-
      🎉 no goals
    -/


instance : Ring s :=
  s.toSubring.toRing


instance : Div s :=
  ⟨fun x y => ⟨x / y, s.div_mem x.2 y.2⟩⟩


instance : Inv s :=
  ⟨fun x => ⟨x⁻¹, s.inv_mem x.2⟩⟩


instance : Pow s ℤ :=
  ⟨fun x z => ⟨x ^ z, s.zpow_mem x.2 z⟩⟩

-- TODO: Those are just special cases of `SubfieldClass.toDivisionRing`/`SubfieldClass.toField`

instance toDivisionRing (s : Subfield K) : DivisionRing s :=
  Subtype.coe_injective.divisionRing ((↑) : s → K) rfl rfl (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl)
    (fun _ ↦ rfl) fun _ ↦ rfl


/-- A subfield inherits a field structure -/
instance toField {K} [Field K] (s : Subfield K) : Field s :=
  Subtype.coe_injective.field ((↑) : s → K) rfl rfl (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ ↦ rfl) (fun _ => rfl)
    (fun _ => rfl) (fun _ ↦ rfl) fun _ => rfl


@[simp, norm_cast]
theorem coe_add (x y : s) : (↑(x + y) : K) = ↑x + ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_sub (x y : s) : (↑(x - y) : K) = ↑x - ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_neg (x : s) : (↑(-x) : K) = -↑x :=
  rfl


@[simp, norm_cast]
theorem coe_mul (x y : s) : (↑(x * y) : K) = ↑x * ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_div (x y : s) : (↑(x / y) : K) = ↑x / ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_inv (x : s) : (↑x⁻¹ : K) = (↑x)⁻¹ :=
  rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : s) : K) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_one : ((1 : s) : K) = 1 :=
  rfl


/-- The embedding from a subfield of the field `K` to `K`. -/
def subtype (s : Subfield K) : s →+* K :=
  { s.toSubmonoid.subtype, s.toAddSubgroup.subtype with toFun := (↑) }


@[simp]
theorem coe_subtype : ⇑(s.subtype) = ((↑) : s → K) :=
  rfl


variable (K) in
theorem toSubring_subtype_eq_subtype (S : Subfield K) :
    S.toSubring.subtype = S.subtype :=
  rfl


theorem mem_toSubmonoid {s : Subfield K} {x : K} : x ∈ s.toSubmonoid ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem coe_toSubmonoid : (s.toSubmonoid : Set K) = s :=
  rfl


@[simp]
theorem mem_toAddSubgroup {s : Subfield K} {x : K} : x ∈ s.toAddSubgroup ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem coe_toAddSubgroup : (s.toAddSubgroup : Set K) = s :=
  rfl


