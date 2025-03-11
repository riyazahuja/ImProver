@[gcongr]
lemma sum_le_sum (h : ∀ i ∈ f.support, h₁ i (f i) ≤ h₂ i (f i)) : f.sum h₁ ≤ f.sum h₂ :=
  Finset.sum_le_sum h


instance instLEFinsupp : LE (ι →₀ α) :=
  ⟨fun f g => ∀ i, f i ≤ g i⟩


lemma le_def : f ≤ g ↔ ∀ i, f i ≤ g i := Iff.rfl


@[simp, norm_cast] lemma coe_le_coe : ⇑f ≤ g ↔ f ≤ g := Iff.rfl


/-- The order on `Finsupp`s over a partial order embeds into the order on functions -/
def orderEmbeddingToFun : (ι →₀ α) ↪o (ι → α) where
  toFun f := f
  inj' f g h :=
    Finsupp.ext fun i => by
      /-
        ι : Type u_1
        κ : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : Zero α
        inst✝ : LE α
        f✝ g✝ f g : Finsupp ι α
        h : Eq ((fun f => ⇑f) f) ((fun f => ⇑f) g)
        i : ι
        ⊢ Eq (f i) (g i)
      -/
      dsimp at h
      /-
        ι : Type u_1
        κ : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : Zero α
        inst✝ : LE α
        f✝ g✝ f g : Finsupp ι α
        h : Eq ⇑f ⇑g
        i : ι
        ⊢ Eq (f i) (g i)
      -/
      rw [h]
      /-
        🎉 no goals
      -/
  map_rel_iff' := coe_le_coe


@[simp]
theorem orderEmbeddingToFun_apply {f : ι →₀ α} {i : ι} : orderEmbeddingToFun f i = f i :=
  rfl


instance preorder : Preorder (ι →₀ α) :=
  { Finsupp.instLEFinsupp with
    le_refl := fun _ _ => le_rfl
    le_trans := fun _ _ _ hfg hgh i => (hfg i).trans (hgh i) }


lemma lt_def : f < g ↔ f ≤ g ∧ ∃ i, f i < g i := Pi.lt_def

@[simp, norm_cast] lemma coe_lt_coe : ⇑f < g ↔ f < g := Iff.rfl


lemma coe_mono : Monotone (Finsupp.toFun : (ι →₀ α) → ι → α) := fun _ _ ↦ id


lemma coe_strictMono : Monotone (Finsupp.toFun : (ι →₀ α) → ι → α) := fun _ _ ↦ id


@[simp] lemma single_le_single : single i a ≤ single i b ↔ a ≤ b := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : Zero α
    inst✝ : Preorder α
    i : ι
    a b : α
    ⊢ Iff (LE.le (Finsupp.single i a) (Finsupp.single i b)) (LE.le a b)
  -/
  classical exact Pi.single_le_single
  /-
    🎉 no goals
  -/


lemma single_mono : Monotone (single i : α → ι →₀ α) := fun _ _ ↦ single_le_single.2


@[gcongr] protected alias ⟨_, GCongr.single_mono⟩ := single_le_single


                                                           /-
                                                             ι : Type u_1
                                                             α : Type u_3
                                                             inst✝¹ : Zero α
                                                             inst✝ : Preorder α
                                                             i : ι
                                                             a : α
                                                             ⊢ Iff (LE.le 0 (Finsupp.single i a)) (LE.le 0 a)
                                                           -/
@[simp] lemma single_nonneg : 0 ≤ single i a ↔ 0 ≤ a := by classical exact Pi.single_nonneg
                                                           /-
                                                             🎉 no goals
                                                           -/

                                                           /-
                                                             ι : Type u_1
                                                             α : Type u_3
                                                             inst✝¹ : Zero α
                                                             inst✝ : Preorder α
                                                             i : ι
                                                             a : α
                                                             ⊢ Iff (LE.le (Finsupp.single i a) 0) (LE.le a 0)
                                                           -/
@[simp] lemma single_nonpos : single i a ≤ 0 ↔ a ≤ 0 := by classical exact Pi.single_nonpos
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma sum_le_sum_index [DecidableEq ι] {f₁ f₂ : ι →₀ α} {h : ι → α → β} (hf : f₁ ≤ f₂)
    (hh : ∀ i ∈ f₁.support ∪ f₂.support, Monotone (h i))
    (hh₀ : ∀ i ∈ f₁.support ∪ f₂.support, h i 0 = 0) : f₁.sum h ≤ f₂.sum h := by
  classical
  rw [sum_of_support_subset _ Finset.subset_union_left _ hh₀,
    sum_of_support_subset _ Finset.subset_union_right _ hh₀]
  exact Finset.sum_le_sum fun i hi ↦ hh _ hi <| hf _


instance partialorder [PartialOrder α] : PartialOrder (ι →₀ α) :=
  { Finsupp.preorder with le_antisymm :=
      fun _f _g hfg hgf => ext fun i => (hfg i).antisymm (hgf i) }


instance semilatticeInf [SemilatticeInf α] : SemilatticeInf (ι →₀ α) :=
  { Finsupp.partialorder with
    inf := zipWith (· ⊓ ·) (inf_idem _)
    inf_le_left := fun _f _g _i => inf_le_left
    inf_le_right := fun _f _g _i => inf_le_right
    le_inf := fun _f _g _i h1 h2 s => le_inf (h1 s) (h2 s) }


@[simp]
theorem inf_apply [SemilatticeInf α] {i : ι} {f g : ι →₀ α} : (f ⊓ g) i = f i ⊓ g i :=
  rfl


instance semilatticeSup [SemilatticeSup α] : SemilatticeSup (ι →₀ α) :=
  { Finsupp.partialorder with
    sup := zipWith (· ⊔ ·) (sup_idem _)
    le_sup_left := fun _f _g _i => le_sup_left
    le_sup_right := fun _f _g _i => le_sup_right
    sup_le := fun _f _g _h hf hg i => sup_le (hf i) (hg i) }


@[simp]
theorem sup_apply [SemilatticeSup α] {i : ι} {f g : ι →₀ α} : (f ⊔ g) i = f i ⊔ g i :=
  rfl


instance lattice [Lattice α] : Lattice (ι →₀ α) :=
  { Finsupp.semilatticeInf, Finsupp.semilatticeSup with }


theorem support_inf_union_support_sup : (f ⊓ g).support ∪ (f ⊔ g).support = f.support ∪ g.support :=
                                         /-
                                           ι : Type u_1
                                           α : Type u_3
                                           inst✝² : Zero α
                                           inst✝¹ : DecidableEq ι
                                           inst✝ : Lattice α
                                           f g : Finsupp ι α
                                           ⊢ Eq (HasCompl.compl ↑(Union.union (Min.min f g).support (Max.max f g).support …
                                         -/
  coe_injective <| compl_injective <| by ext; simp [inf_eq_and_sup_eq_iff]
                                              /-
                                                🎉 no goals
                                              -/


theorem support_sup_union_support_inf : (f ⊔ g).support ∪ (f ⊓ g).support = f.support ∪ g.support :=
  (union_comm _ _).trans <| support_inf_union_support_sup _ _


instance orderedAddCommMonoid : OrderedAddCommMonoid (ι →₀ α) :=
  { Finsupp.instAddCommMonoid, Finsupp.partialorder with
    add_le_add_left := fun _a _b h c s => add_le_add_left (h s) (c s) }


lemma mapDomain_mono : Monotone (mapDomain f : (ι →₀ α) → (κ →₀ α)) := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝ : OrderedAddCommMonoid α
    f : ι → κ
    ⊢ Monotone (Finsupp.mapDomain f)
  -/
  classical exact fun g₁ g₂ h ↦ sum_le_sum_index h (fun _ _ ↦ single_mono) (by simp)
  /-
    🎉 no goals
  -/


@[gcongr] protected lemma GCongr.mapDomain_mono (hg : g₁ ≤ g₂) : g₁.mapDomain f ≤ g₂.mapDomain f :=
  mapDomain_mono hg


                                                              /-
                                                                ι : Type u_1
                                                                κ : Type u_2
                                                                α : Type u_3
                                                                inst✝ : OrderedAddCommMonoid α
                                                                f : ι → κ
                                                                g : Finsupp ι α
                                                                hg : LE.le 0 g
                                                                ⊢ LE.le 0 (Finsupp.mapDomain f g)
                                                              -/
lemma mapDomain_nonneg (hg : 0 ≤ g) : 0 ≤ g.mapDomain f := by simpa using mapDomain_mono hg
                                                              /-
                                                                🎉 no goals
                                                              -/

                                                              /-
                                                                ι : Type u_1
                                                                κ : Type u_2
                                                                α : Type u_3
                                                                inst✝ : OrderedAddCommMonoid α
                                                                f : ι → κ
                                                                g : Finsupp ι α
                                                                hg : LE.le g 0
                                                                ⊢ LE.le (Finsupp.mapDomain f g) 0
                                                              -/
lemma mapDomain_nonpos (hg : g ≤ 0) : g.mapDomain f ≤ 0 := by simpa using mapDomain_mono hg
                                                              /-
                                                                🎉 no goals
                                                              -/


instance orderedCancelAddCommMonoid [OrderedCancelAddCommMonoid α] :
    OrderedCancelAddCommMonoid (ι →₀ α) :=
  { Finsupp.orderedAddCommMonoid with
    le_of_add_le_add_left := fun _f _g _i h s => le_of_add_le_add_left (h s) }


instance addLeftReflectLE [OrderedAddCommMonoid α] [AddLeftReflectLE α] :
    AddLeftReflectLE (ι →₀ α) :=
  ⟨fun _f _g _h H x => le_of_add_le_add_left <| H x⟩


instance instPosSMulMono [PosSMulMono α β] : PosSMulMono α (ι →₀ β) :=
  PosSMulMono.lift _ coe_le_coe coe_smul


instance instSMulPosMono [SMulPosMono α β] : SMulPosMono α (ι →₀ β) :=
  SMulPosMono.lift _ coe_le_coe coe_smul coe_zero


instance instPosSMulReflectLE [PosSMulReflectLE α β] : PosSMulReflectLE α (ι →₀ β) :=
  PosSMulReflectLE.lift _ coe_le_coe coe_smul


instance instSMulPosReflectLE [SMulPosReflectLE α β] : SMulPosReflectLE α (ι →₀ β) :=
  SMulPosReflectLE.lift _ coe_le_coe coe_smul coe_zero


instance instPosSMulStrictMono [PosSMulStrictMono α β] : PosSMulStrictMono α (ι →₀ β) :=
  PosSMulStrictMono.lift _ coe_le_coe coe_smul


instance instSMulPosStrictMono [SMulPosStrictMono α β] : SMulPosStrictMono α (ι →₀ β) :=
  SMulPosStrictMono.lift _ coe_le_coe coe_smul coe_zero

-- `PosSMulReflectLT α (ι →₀ β)` already follows from the other instances


instance instSMulPosReflectLT [SMulPosReflectLT α β] : SMulPosReflectLT α (ι →₀ β) :=
  SMulPosReflectLT.lift _ coe_le_coe coe_smul coe_zero


instance orderBot : OrderBot (ι →₀ α) where
  bot := 0
               /-
                 ι : Type u_1
                 κ : Type u_2
                 α : Type u_3
                 β : Type u_4
                 inst✝ : CanonicallyOrderedAddCommMonoid α
                 f g : Finsupp ι α
                 ⊢ ∀ (a : Finsupp ι α), LE.le Bot.bot a
               -/
  bot_le := by simp only [le_def, coe_zero, Pi.zero_apply, imp_true_iff, zero_le]
               /-
                 🎉 no goals
               -/


protected theorem bot_eq_zero : (⊥ : ι →₀ α) = 0 :=
  rfl


@[simp]
theorem add_eq_zero_iff (f g : ι →₀ α) : f + g = 0 ↔ f = 0 ∧ g = 0 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝ : CanonicallyOrderedAddCommMonoid α
    f g : Finsupp ι α
    ⊢ Iff (Eq (HAdd.hAdd f g) 0) (And (Eq f 0) (Eq g 0))
  -/
  simp [DFunLike.ext_iff, forall_and]
  /-
    🎉 no goals
  -/


theorem le_iff' (f g : ι →₀ α) {s : Finset ι} (hf : f.support ⊆ s) : f ≤ g ↔ ∀ i ∈ s, f i ≤ g i :=
  ⟨fun h s _hs => h s, fun h s => by
    classical exact
        if H : s ∈ f.support then h s (hf H) else (not_mem_support_iff.1 H).symm ▸ zero_le (g s)⟩


theorem le_iff (f g : ι →₀ α) : f ≤ g ↔ ∀ i ∈ f.support, f i ≤ g i :=
  le_iff' f g <| Subset.refl _


lemma support_monotone : Monotone (support (α := ι) (M := α)) :=
                      /-
                        ι : Type u_1
                        α : Type u_3
                        inst✝ : CanonicallyOrderedAddCommMonoid α
                        f g : Finsupp ι α
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


instance decidableLE [DecidableRel (@LE.le α _)] : DecidableRel (@LE.le (ι →₀ α) _) := fun f g =>
  decidable_of_iff _ (le_iff f g).symm


instance decidableLT [DecidableRel (@LE.le α _)] : DecidableRel (@LT.lt (ι →₀ α) _) :=
  decidableLTOfDecidableLE


@[simp]
theorem single_le_iff {i : ι} {x : α} {f : ι →₀ α} : single i x ≤ f ↔ x ≤ f i :=
                                                  /-
                                                    ι : Type u_1
                                                    α : Type u_3
                                                    inst✝ : CanonicallyOrderedAddCommMonoid α
                                                    i : ι
                                                    x : α
                                                    f : Finsupp ι α
                                                    ⊢ Iff (∀ (i_1 : ι), Membership.mem (Singleton.singleton i) i_1 → LE.le ((Finsu …
                                                  -/
  (le_iff' _ _ support_single_subset).trans <| by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- This is called `tsub` for truncated subtraction, to distinguish it with subtraction in an
additive group. -/
instance tsub : Sub (ι →₀ α) :=
  ⟨zipWith (fun m n => m - n) (tsub_self 0)⟩


instance orderedSub : OrderedSub (ι →₀ α) :=
  ⟨fun _n _m _k => forall_congr' fun _x => tsub_le_iff_right⟩


instance : CanonicallyOrderedAddCommMonoid (ι →₀ α) :=
  { Finsupp.orderBot,
    Finsupp.orderedAddCommMonoid with
    exists_add_of_le := fun {f g} h => ⟨g - f, ext fun x => (add_tsub_cancel_of_le <| h x).symm⟩
    le_self_add := fun _f _g _x => le_self_add }


@[simp, norm_cast] lemma coe_tsub (f g : ι →₀ α) : ⇑(f - g) = f - g := rfl


theorem tsub_apply (f g : ι →₀ α) (a : ι) : (f - g) a = f a - g a :=
  rfl


@[simp]
theorem single_tsub : single i (a - b) = single i a - single i b := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    i : ι
    a b : α
    ⊢ Eq (Finsupp.single i (HSub.hSub a b)) (HSub.hSub (Finsupp.single i a) (Finsu …
  -/
  ext j
  /-
    case h
    ι : Type u_1
    α : Type u_3
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    i : ι
    a b : α
    j : ι
    ⊢ Eq ((Finsupp.single i (HSub.hSub a b)) j) ((HSub.hSub (Finsupp.single i a) ( …
  -/
  obtain rfl | h := eq_or_ne i j
    /-
      case h.inl
      ι : Type u_1
      α : Type u_3
      inst✝² : CanonicallyOrderedAddCommMonoid α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      i : ι
      a b : α
      ⊢ Eq ((Finsupp.single i (HSub.hSub a b)) i) ((HSub.hSub (Finsupp.single i a) ( …
    -/
  · rw [tsub_apply, single_eq_same, single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u_1
      α : Type u_3
      inst✝² : CanonicallyOrderedAddCommMonoid α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      i : ι
      a b : α
      j : ι
      h : Ne i j
      ⊢ Eq ((Finsupp.single i (HSub.hSub a b)) j) ((HSub.hSub (Finsupp.single i a) ( …
    -/
  · rw [tsub_apply, single_eq_of_ne h, single_eq_of_ne h, single_eq_of_ne h, tsub_self]
    /-
      🎉 no goals
    -/


theorem support_tsub {f1 f2 : ι →₀ α} : (f1 - f2).support ⊆ f1.support := by
  simp +contextual only [subset_iff, tsub_eq_zero_iff_le, mem_support_iff,
    Ne, coe_tsub, Pi.sub_apply, not_imp_not, zero_le, imp_true_iff]


theorem subset_support_tsub [DecidableEq ι] {f1 f2 : ι →₀ α} :
    f1.support \ f2.support ⊆ (f1 - f2).support := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝³ : CanonicallyOrderedAddCommMonoid α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : DecidableEq ι
    f1 f2 : Finsupp ι α
    ⊢ HasSubset.Subset (SDiff.sdiff f1.support f2.support) (HSub.hSub f1 f2).support
  -/
  simp +contextual [subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_inf [DecidableEq ι] (f g : ι →₀ α) : (f ⊓ g).support = f.support ∩ g.support := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CanonicallyLinearOrderedAddCommMonoid α
    inst✝ : DecidableEq ι
    f g : Finsupp ι α
    ⊢ Eq (Min.min f g).support (Inter.inter f.support g.support)
  -/
  ext
  simp only [inf_apply, mem_support_iff, Ne, Finset.mem_union, Finset.mem_filter,
    Finset.mem_inter]
  /-
    case h
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CanonicallyLinearOrderedAddCommMonoid α
    inst✝ : DecidableEq ι
    f g : Finsupp ι α
    a✝ : ι
    ⊢ Iff (Not (Eq (Min.min (f a✝) (g a✝)) 0)) (And (Not (Eq (f a✝) 0)) (Not (Eq ( …
  -/
  simp only [← nonpos_iff_eq_zero, min_le_iff, not_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_sup [DecidableEq ι] (f g : ι →₀ α) : (f ⊔ g).support = f.support ∪ g.support := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CanonicallyLinearOrderedAddCommMonoid α
    inst✝ : DecidableEq ι
    f g : Finsupp ι α
    ⊢ Eq (Max.max f g).support (Union.union f.support g.support)
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CanonicallyLinearOrderedAddCommMonoid α
    inst✝ : DecidableEq ι
    f g : Finsupp ι α
    a✝ : ι
    ⊢ Iff (Membership.mem (Max.max f g).support a✝) (Membership.mem (Union.union f …
  -/
  simp only [Finset.mem_union, mem_support_iff, sup_apply, Ne, ← bot_eq_zero]
  /-
    case h
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CanonicallyLinearOrderedAddCommMonoid α
    inst✝ : DecidableEq ι
    f g : Finsupp ι α
    a✝ : ι
    ⊢ Iff (Not (Eq (Max.max (f a✝) (g a✝)) Bot.bot)) (Or (Not (Eq (f a✝) Bot.bot)) …
  -/
  rw [_root_.sup_eq_bot_iff, not_and_or]
  /-
    🎉 no goals
  -/


nonrec theorem disjoint_iff {f g : ι →₀ α} : Disjoint f g ↔ Disjoint f.support g.support := by
  classical
    rw [disjoint_iff, disjoint_iff, Finsupp.bot_eq_zero, ← Finsupp.support_eq_empty,
      Finsupp.support_inf]
    rfl


theorem sub_single_one_add {a : ι} {u u' : ι →₀ ℕ} (h : u a ≠ 0) :
    u - single a 1 + u' = u + u' - single a 1 :=
  tsub_add_eq_add_tsub <| single_le_iff.mpr <| Nat.one_le_iff_ne_zero.mpr h


theorem add_sub_single_one {a : ι} {u u' : ι →₀ ℕ} (h : u' a ≠ 0) :
    u + (u' - single a 1) = u + u' - single a 1 :=
  (add_tsub_assoc_of_le (single_le_iff.mpr <| Nat.one_le_iff_ne_zero.mpr h) _).symm


lemma sub_add_single_one_cancel {u : ι →₀ ℕ} {i : ι} (h : u i ≠ 0) :
    u - single i 1 + single i 1 = u := by
  /-
    ι : Type u_1
    u : Finsupp ι Nat
    i : ι
    h : Ne (u i) 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub u (Finsupp.single i 1)) (Finsupp.single i 1)) u
  -/
  rw [sub_single_one_add h, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


