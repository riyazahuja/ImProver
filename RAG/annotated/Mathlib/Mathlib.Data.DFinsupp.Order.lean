instance : LE (Π₀ i, α i) :=
  ⟨fun f g ↦ ∀ i, f i ≤ g i⟩


lemma le_def : f ≤ g ↔ ∀ i, f i ≤ g i := Iff.rfl


@[simp, norm_cast] lemma coe_le_coe : ⇑f ≤ g ↔ f ≤ g := Iff.rfl


/-- The order on `DFinsupp`s over a partial order embeds into the order on functions -/
def orderEmbeddingToFun : (Π₀ i, α i) ↪o ∀ i, α i where
  toFun := DFunLike.coe
  inj' := DFunLike.coe_injective
  map_rel_iff' :=
    #adaptation_note
    /--
    This proof used to be `rfl`,
    but has been temporarily broken by https://github.com/leanprover/lean4/pull/5329.
    It can hopefully be restored after https://github.com/leanprover/lean4/pull/5359
    -/
    Iff.rfl


@[simp, norm_cast]
lemma coe_orderEmbeddingToFun : ⇑(orderEmbeddingToFun (α := α)) = DFunLike.coe := rfl

-- Porting note: we added implicit arguments here in https://github.com/leanprover-community/mathlib4/pull/3414.

theorem orderEmbeddingToFun_apply {f : Π₀ i, α i} {i : ι} :
    (@orderEmbeddingToFun ι α _ _ f) i = f i :=
  rfl


instance : Preorder (Π₀ i, α i) :=
  { (inferInstance : LE (DFinsupp α)) with
    le_refl := fun _ _ ↦ le_rfl
    le_trans := fun _ _ _ hfg hgh i ↦ (hfg i).trans (hgh i) }


lemma lt_def : f < g ↔ f ≤ g ∧ ∃ i, f i < g i := Pi.lt_def

@[simp, norm_cast] lemma coe_lt_coe : ⇑f < g ↔ f < g := Iff.rfl


lemma coe_mono : Monotone ((⇑) : (Π₀ i, α i) → ∀ i, α i) := fun _ _ ↦ id


lemma coe_strictMono : Monotone ((⇑) : (Π₀ i, α i) → ∀ i, α i) := fun _ _ ↦ id


instance [∀ i, PartialOrder (α i)] : PartialOrder (Π₀ i, α i) :=
  { (inferInstance : Preorder (DFinsupp α)) with
    le_antisymm := fun _ _ hfg hgf ↦ ext fun i ↦ (hfg i).antisymm (hgf i) }


instance [∀ i, SemilatticeInf (α i)] : SemilatticeInf (Π₀ i, α i) :=
  { (inferInstance : PartialOrder (DFinsupp α)) with
    inf := zipWith (fun _ ↦ (· ⊓ ·)) fun _ ↦ inf_idem _
    inf_le_left := fun _ _ _ ↦ inf_le_left
    inf_le_right := fun _ _ _ ↦ inf_le_right
    le_inf := fun _ _ _ hf hg i ↦ le_inf (hf i) (hg i) }


@[simp, norm_cast]
lemma coe_inf [∀ i, SemilatticeInf (α i)] (f g : Π₀ i, α i) : f ⊓ g = ⇑f ⊓ g := rfl


theorem inf_apply [∀ i, SemilatticeInf (α i)] (f g : Π₀ i, α i) (i : ι) : (f ⊓ g) i = f i ⊓ g i :=
  zipWith_apply _ _ _ _ _


instance [∀ i, SemilatticeSup (α i)] : SemilatticeSup (Π₀ i, α i) :=
  { (inferInstance : PartialOrder (DFinsupp α)) with
    sup := zipWith (fun _ ↦ (· ⊔ ·)) fun _ ↦ sup_idem _
    le_sup_left := fun _ _ _ ↦ le_sup_left
    le_sup_right := fun _ _ _ ↦ le_sup_right
    sup_le := fun _ _ _ hf hg i ↦ sup_le (hf i) (hg i) }


@[simp, norm_cast]
lemma coe_sup [∀ i, SemilatticeSup (α i)] (f g : Π₀ i, α i) : f ⊔ g = ⇑f ⊔ g := rfl


theorem sup_apply [∀ i, SemilatticeSup (α i)] (f g : Π₀ i, α i) (i : ι) : (f ⊔ g) i = f i ⊔ g i :=
  zipWith_apply _ _ _ _ _


instance lattice : Lattice (Π₀ i, α i) :=
  { (inferInstance : SemilatticeInf (DFinsupp α)),
    (inferInstance : SemilatticeSup (DFinsupp α)) with }


theorem support_inf_union_support_sup : (f ⊓ g).support ∪ (f ⊔ g).support = f.support ∪ g.support :=
                                         /-
                                           ι : Type u_1
                                           α : ι → Type u_2
                                           inst✝³ : (i : ι) → Zero (α i)
                                           inst✝² : (i : ι) → Lattice (α i)
                                           f g : DFinsupp fun i => α i
                                           inst✝¹ : DecidableEq ι
                                           inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
                                           ⊢ Eq (HasCompl.compl ↑(Union.union (Min.min f g).support (Max.max f g).support …
                                         -/
  coe_injective <| compl_injective <| by ext; simp [inf_eq_and_sup_eq_iff]
                                              /-
                                                🎉 no goals
                                              -/


theorem support_sup_union_support_inf : (f ⊔ g).support ∪ (f ⊓ g).support = f.support ∪ g.support :=
  (union_comm _ _).trans <| support_inf_union_support_sup _ _


instance (α : ι → Type*) [∀ i, OrderedAddCommMonoid (α i)] : OrderedAddCommMonoid (Π₀ i, α i) :=
  { (inferInstance : AddCommMonoid (DFinsupp α)),
    (inferInstance : PartialOrder (DFinsupp α)) with
    add_le_add_left := fun _ _ h c i ↦ add_le_add_left (h i) (c i) }


instance (α : ι → Type*) [∀ i, OrderedCancelAddCommMonoid (α i)] :
    OrderedCancelAddCommMonoid (Π₀ i, α i) :=
  { (inferInstance : OrderedAddCommMonoid (DFinsupp α)) with
    le_of_add_le_add_left := fun _ _ _ H i ↦ le_of_add_le_add_left (H i) }


instance [∀ i, OrderedAddCommMonoid (α i)] [∀ i, AddLeftReflectLE (α i)] :
    AddLeftReflectLE (Π₀ i, α i) :=
  ⟨fun _ _ _ H i ↦ le_of_add_le_add_left (H i)⟩


instance instPosSMulMono [∀ i, PosSMulMono α (β i)] : PosSMulMono α (Π₀ i, β i) :=
  PosSMulMono.lift _ coe_le_coe coe_smul


instance instSMulPosMono [∀ i, SMulPosMono α (β i)] : SMulPosMono α (Π₀ i, β i) :=
  SMulPosMono.lift _ coe_le_coe coe_smul coe_zero


instance instPosSMulReflectLE [∀ i, PosSMulReflectLE α (β i)] : PosSMulReflectLE α (Π₀ i, β i) :=
  PosSMulReflectLE.lift _ coe_le_coe coe_smul


instance instSMulPosReflectLE [∀ i, SMulPosReflectLE α (β i)] : SMulPosReflectLE α (Π₀ i, β i) :=
  SMulPosReflectLE.lift _ coe_le_coe coe_smul coe_zero


instance instPosSMulStrictMono [∀ i, PosSMulStrictMono α (β i)] : PosSMulStrictMono α (Π₀ i, β i) :=
  PosSMulStrictMono.lift _ coe_le_coe coe_smul


instance instSMulPosStrictMono [∀ i, SMulPosStrictMono α (β i)] : SMulPosStrictMono α (Π₀ i, β i) :=
  SMulPosStrictMono.lift _ coe_le_coe coe_smul coe_zero

-- Note: There is no interesting instance for `PosSMulReflectLT α (Π₀ i, β i)` that's not already
-- implied by the other instances


instance instSMulPosReflectLT [∀ i, SMulPosReflectLT α (β i)] : SMulPosReflectLT α (Π₀ i, β i) :=
  SMulPosReflectLT.lift _ coe_le_coe coe_smul coe_zero


instance : OrderBot (Π₀ i, α i) where
  bot := 0
               /-
                 ι : Type u_1
                 α : ι → Type u_2
                 inst✝ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
                 ⊢ ∀ (a : DFinsupp fun i => α i), LE.le Bot.bot a
               -/
  bot_le := by simp only [le_def, coe_zero, Pi.zero_apply, imp_true_iff, zero_le]
               /-
                 🎉 no goals
               -/


protected theorem bot_eq_zero : (⊥ : Π₀ i, α i) = 0 :=
  rfl


@[simp]
theorem add_eq_zero_iff (f g : Π₀ i, α i) : f + g = 0 ↔ f = 0 ∧ g = 0 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    f g : DFinsupp fun i => α i
    ⊢ Iff (Eq (HAdd.hAdd f g) 0) (And (Eq f 0) (Eq g 0))
  -/
  simp [DFunLike.ext_iff, forall_and]
  /-
    🎉 no goals
  -/


theorem le_iff' (hf : f.support ⊆ s) : f ≤ g ↔ ∀ i ∈ s, f i ≤ g i :=
  ⟨fun h s _ ↦ h s, fun h s ↦
    if H : s ∈ f.support then h s (hf H) else (not_mem_support_iff.1 H).symm ▸ zero_le (g s)⟩


theorem le_iff : f ≤ g ↔ ∀ i ∈ f.support, f i ≤ g i :=
  le_iff' <| Subset.refl _


lemma support_monotone : Monotone (support (ι := ι) (β := α)) :=
                      /-
                        ι : Type u_1
                        α : ι → Type u_2
                        inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
                        inst✝¹ : DecidableEq ι
                        inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
                        f g : DFinsupp fun i => α i
                        h : LE.le f g
                        a : ι
                        ha : Membership.mem f.support a
                        ⊢ Membership.mem g.support a
                      -/
  fun f g h a ha ↦ by rw [mem_support_iff, ← pos_iff_ne_zero] at ha ⊢; exact ha.trans_le (h _)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma support_mono (hfg : f ≤ g) : f.support ⊆ g.support := support_monotone hfg


instance decidableLE [∀ i, DecidableRel (@LE.le (α i) _)] : DecidableRel (@LE.le (Π₀ i, α i) _) :=
  fun _ _ ↦ decidable_of_iff _ le_iff.symm


@[simp]
theorem single_le_iff {f : Π₀ i, α i} {i : ι} {a : α i} : single i a ≤ f ↔ a ≤ f i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => α i
    i : ι
    a : α i
    ⊢ Iff (LE.le (DFinsupp.single i a) f) (LE.le a (f i))
  -/
  classical exact (le_iff' support_single_subset).trans <| by simp
  /-
    🎉 no goals
  -/


/-- This is called `tsub` for truncated subtraction, to distinguish it with subtraction in an
additive group. -/
instance tsub : Sub (Π₀ i, α i) :=
  ⟨zipWith (fun _ m n ↦ m - n) fun _ ↦ tsub_self 0⟩


theorem tsub_apply (f g : Π₀ i, α i) (i : ι) : (f - g) i = f i - g i :=
  zipWith_apply _ _ _ _ _


@[simp, norm_cast]
theorem coe_tsub (f g : Π₀ i, α i) : ⇑(f - g) = f - g := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝¹ : (i : ι) → Sub (α i)
    inst✝ : ∀ (i : ι), OrderedSub (α i)
    f g : DFinsupp fun i => α i
    ⊢ Eq (⇑(HSub.hSub f g)) (HSub.hSub ⇑f ⇑g)
  -/
  ext i
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝¹ : (i : ι) → Sub (α i)
    inst✝ : ∀ (i : ι), OrderedSub (α i)
    f g : DFinsupp fun i => α i
    i : ι
    ⊢ Eq ((HSub.hSub f g) i) (HSub.hSub (⇑f) (⇑g) i)
  -/
  exact tsub_apply f g i
  /-
    🎉 no goals
  -/


instance : OrderedSub (Π₀ i, α i) :=
  ⟨fun _ _ _ ↦ forall_congr' fun _ ↦ tsub_le_iff_right⟩


instance : CanonicallyOrderedAddCommMonoid (Π₀ i, α i) :=
  { (inferInstance : OrderBot (DFinsupp α)),
    (inferInstance : OrderedAddCommMonoid (DFinsupp α)) with
    exists_add_of_le := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
        inst✝¹ : (i : ι) → Sub (α i)
        inst✝ : ∀ (i : ι), OrderedSub (α i)
        f g : DFinsupp fun i => α i
        i : ι
        a b : α i
        ⊢ ∀ {a b : DFinsupp fun i => α i}, LE.le a b → Exists fun c => Eq b (HAdd.hAdd …
      -/
      intro f g h
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
        inst✝¹ : (i : ι) → Sub (α i)
        inst✝ : ∀ (i : ι), OrderedSub (α i)
        f✝ g✝ : DFinsupp fun i => α i
        i : ι
        a b : α i
        f g : DFinsupp fun i => α i
        h : LE.le f g
        ⊢ Exists fun c => Eq g (HAdd.hAdd f c)
      -/
      exists g - f
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
        inst✝¹ : (i : ι) → Sub (α i)
        inst✝ : ∀ (i : ι), OrderedSub (α i)
        f✝ g✝ : DFinsupp fun i => α i
        i : ι
        a b : α i
        f g : DFinsupp fun i => α i
        h : LE.le f g
        ⊢ Eq g (HAdd.hAdd f (HSub.hSub g f))
      -/
      ext i
      /-
        case h
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
        inst✝¹ : (i : ι) → Sub (α i)
        inst✝ : ∀ (i : ι), OrderedSub (α i)
        f✝ g✝ : DFinsupp fun i => α i
        i✝ : ι
        a b : α i✝
        f g : DFinsupp fun i => α i
        h : LE.le f g
        i : ι
        ⊢ Eq (g i) ((HAdd.hAdd f (HSub.hSub g f)) i)
      -/
      exact (add_tsub_cancel_of_le <| h i).symm
      /-
        🎉 no goals
      -/
    le_self_add := fun _ _ _ ↦ le_self_add }


@[simp]
theorem single_tsub : single i (a - b) = single i a - single i b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝³ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝² : (i : ι) → Sub (α i)
    inst✝¹ : ∀ (i : ι), OrderedSub (α i)
    i : ι
    a b : α i
    inst✝ : DecidableEq ι
    ⊢ Eq (DFinsupp.single i (HSub.hSub a b)) (HSub.hSub (DFinsupp.single i a) (DFi …
  -/
  ext j
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝³ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝² : (i : ι) → Sub (α i)
    inst✝¹ : ∀ (i : ι), OrderedSub (α i)
    i : ι
    a b : α i
    inst✝ : DecidableEq ι
    j : ι
    ⊢ Eq ((DFinsupp.single i (HSub.hSub a b)) j) ((HSub.hSub (DFinsupp.single i a) …
  -/
  obtain rfl | h := eq_or_ne i j
    /-
      case h.inl
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
      inst✝² : (i : ι) → Sub (α i)
      inst✝¹ : ∀ (i : ι), OrderedSub (α i)
      i : ι
      a b : α i
      inst✝ : DecidableEq ι
      ⊢ Eq ((DFinsupp.single i (HSub.hSub a b)) i) ((HSub.hSub (DFinsupp.single i a) …
    -/
  · rw [tsub_apply, single_eq_same, single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
      inst✝² : (i : ι) → Sub (α i)
      inst✝¹ : ∀ (i : ι), OrderedSub (α i)
      i : ι
      a b : α i
      inst✝ : DecidableEq ι
      j : ι
      h : Ne i j
      ⊢ Eq ((DFinsupp.single i (HSub.hSub a b)) j) ((HSub.hSub (DFinsupp.single i a) …
    -/
  · rw [tsub_apply, single_eq_of_ne h, single_eq_of_ne h, single_eq_of_ne h, tsub_self]
    /-
      🎉 no goals
    -/


theorem support_tsub : (f - g).support ⊆ f.support := by
  simp +contextual only [subset_iff, tsub_eq_zero_iff_le, mem_support_iff,
    Ne, coe_tsub, Pi.sub_apply, not_imp_not, zero_le, imp_true_iff]


theorem subset_support_tsub : f.support \ g.support ⊆ (f - g).support := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝³ : (i : ι) → Sub (α i)
    inst✝² : ∀ (i : ι), OrderedSub (α i)
    f g : DFinsupp fun i => α i
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
    ⊢ HasSubset.Subset (SDiff.sdiff f.support g.support) (HSub.hSub f g).support
  -/
  simp +contextual [subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_inf : (f ⊓ g).support = f.support ∩ g.support := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    ⊢ Eq (Min.min f g).support (Inter.inter f.support g.support)
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    a✝ : ι
    ⊢ Iff (Membership.mem (Min.min f g).support a✝) (Membership.mem (Inter.inter f …
  -/
  simp only [inf_apply, mem_support_iff, Ne, Finset.mem_inter]
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    a✝ : ι
    ⊢ Iff (Not (Eq (Min.min (f a✝) (g a✝)) 0)) (And (Not (Eq (f a✝) 0)) (Not (Eq ( …
  -/
  simp only [← nonpos_iff_eq_zero, min_le_iff, not_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_sup : (f ⊔ g).support = f.support ∪ g.support := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    ⊢ Eq (Max.max f g).support (Union.union f.support g.support)
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    a✝ : ι
    ⊢ Iff (Membership.mem (Max.max f g).support a✝) (Membership.mem (Union.union f …
  -/
  simp only [Finset.mem_union, mem_support_iff, sup_apply, Ne, ← bot_eq_zero]
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    a✝ : ι
    ⊢ Iff (Not (Eq (Max.max (f a✝) (g a✝)) Bot.bot)) (Or (Not (Eq (f a✝) Bot.bot)) …
  -/
  rw [_root_.sup_eq_bot_iff, not_and_or]
  /-
    🎉 no goals
  -/


nonrec theorem disjoint_iff : Disjoint f g ↔ Disjoint f.support g.support := by
  rw [disjoint_iff, disjoint_iff, DFinsupp.bot_eq_zero, ← DFinsupp.support_eq_empty,
    DFinsupp.support_inf]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → CanonicallyLinearOrderedAddCommMonoid (α i)
    inst✝ : DecidableEq ι
    f g : DFinsupp fun i => α i
    ⊢ Iff (Eq (Inter.inter f.support g.support) EmptyCollection.emptyCollection) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


