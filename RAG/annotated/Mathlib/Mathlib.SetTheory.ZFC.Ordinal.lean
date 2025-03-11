/-- A transitive set is one where every element is a subset.

This is equivalent to being an infinite-open interval in the transitive closure of membership. -/
def IsTransitive (x : ZFSet) : Prop :=
  ∀ y ∈ x, y ⊆ x


@[simp]
theorem isTransitive_empty : IsTransitive ∅ := fun y hy => (not_mem_empty y hy).elim


@[deprecated isTransitive_empty (since := "2024-09-21")]
alias empty_isTransitive := isTransitive_empty


theorem IsTransitive.subset_of_mem (h : x.IsTransitive) : y ∈ x → y ⊆ x := h y


theorem isTransitive_iff_mem_trans : z.IsTransitive ↔ ∀ {x y : ZFSet}, x ∈ y → y ∈ z → x ∈ z :=
  ⟨fun h _ _ hx hy => h.subset_of_mem hy hx, fun H _ hx _ hy => H hy hx⟩


alias ⟨IsTransitive.mem_trans, _⟩ := isTransitive_iff_mem_trans


protected theorem IsTransitive.inter (hx : x.IsTransitive) (hy : y.IsTransitive) :
    (x ∩ y).IsTransitive := fun z hz w hw => by
  /-
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    z : ZFSet.{u}
    hz : Membership.mem (Inter.inter x y) z
    w : ZFSet.{u}
    hw : Membership.mem z w
    ⊢ Membership.mem (Inter.inter x y) w
  -/
  rw [mem_inter] at hz ⊢
  /-
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    z : ZFSet.{u}
    hz : And (Membership.mem x z) (Membership.mem y z)
    w : ZFSet.{u}
    hw : Membership.mem z w
    ⊢ And (Membership.mem x w) (Membership.mem y w)
  -/
  exact ⟨hx.mem_trans hw hz.1, hy.mem_trans hw hz.2⟩
  /-
    🎉 no goals
  -/


/-- The union of a transitive set is transitive. -/
protected theorem IsTransitive.sUnion (h : x.IsTransitive) :
    (⋃₀ x : ZFSet).IsTransitive := fun y hy z hz => by
  /-
    x : ZFSet.{u}
    h : x.IsTransitive
    y : ZFSet.{u}
    hy : Membership.mem x.sUnion y
    z : ZFSet.{u}
    hz : Membership.mem y z
    ⊢ Membership.mem x.sUnion z
  -/
  rcases mem_sUnion.1 hy with ⟨w, hw, hw'⟩
  /-
    case intro.intro
    x : ZFSet.{u}
    h : x.IsTransitive
    y : ZFSet.{u}
    hy : Membership.mem x.sUnion y
    z : ZFSet.{u}
    hz : Membership.mem y z
    w : ZFSet.{u}
    hw : Membership.mem x w
    hw' : Membership.mem w y
    ⊢ Membership.mem x.sUnion z
  -/
  exact mem_sUnion_of_mem hz (h.mem_trans hw' hw)
  /-
    🎉 no goals
  -/


/-- The union of transitive sets is transitive. -/
theorem IsTransitive.sUnion' (H : ∀ y ∈ x, IsTransitive y) :
    (⋃₀ x : ZFSet).IsTransitive := fun y hy z hz => by
  /-
    x : ZFSet.{u}
    H : ∀ (y : ZFSet.{u}), Membership.mem x y → y.IsTransitive
    y : ZFSet.{u}
    hy : Membership.mem x.sUnion y
    z : ZFSet.{u}
    hz : Membership.mem y z
    ⊢ Membership.mem x.sUnion z
  -/
  rcases mem_sUnion.1 hy with ⟨w, hw, hw'⟩
  /-
    case intro.intro
    x : ZFSet.{u}
    H : ∀ (y : ZFSet.{u}), Membership.mem x y → y.IsTransitive
    y : ZFSet.{u}
    hy : Membership.mem x.sUnion y
    z : ZFSet.{u}
    hz : Membership.mem y z
    w : ZFSet.{u}
    hw : Membership.mem x w
    hw' : Membership.mem w y
    ⊢ Membership.mem x.sUnion z
  -/
  exact mem_sUnion_of_mem ((H w hw).mem_trans hz hw') hw
  /-
    🎉 no goals
  -/


protected theorem IsTransitive.union (hx : x.IsTransitive) (hy : y.IsTransitive) :
    (x ∪ y).IsTransitive := by
  /-
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    ⊢ (Union.union x y).IsTransitive
  -/
  rw [← sUnion_pair]
  /-
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    ⊢ (Insert.insert x (Singleton.singleton y)).sUnion.IsTransitive
  -/
  apply IsTransitive.sUnion'
  /-
    case H
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    ⊢ ∀ (y_1 : ZFSet.{u}), Membership.mem (Insert.insert x (Singleton.singleton y) …
  -/
  intro
  /-
    case H
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    y✝ : ZFSet.{u}
    ⊢ Membership.mem (Insert.insert x (Singleton.singleton y)) y✝ → y✝.IsTransitive
  -/
  rw [mem_pair]
  /-
    case H
    x y : ZFSet.{u}
    hx : x.IsTransitive
    hy : y.IsTransitive
    y✝ : ZFSet.{u}
    ⊢ Or (Eq y✝ x) (Eq y✝ y) → y✝.IsTransitive
  -/
  rintro (rfl | rfl)
  /-
    case H.inl
    y : ZFSet.{u}
    hy : y.IsTransitive
    y✝ : ZFSet.{u}
    hx : y✝.IsTransitive
    ⊢ y✝.IsTransitive
  -/
  assumption'
  /-
    🎉 no goals
  -/


protected theorem IsTransitive.powerset (h : x.IsTransitive) : (powerset x).IsTransitive :=
  fun y hy z hz => by
  /-
    x : ZFSet.{u}
    h : x.IsTransitive
    y : ZFSet.{u}
    hy : Membership.mem x.powerset y
    z : ZFSet.{u}
    hz : Membership.mem y z
    ⊢ Membership.mem x.powerset z
  -/
  rw [mem_powerset] at hy ⊢
  /-
    x : ZFSet.{u}
    h : x.IsTransitive
    y : ZFSet.{u}
    hy : HasSubset.Subset y x
    z : ZFSet.{u}
    hz : Membership.mem y z
    ⊢ HasSubset.Subset z x
  -/
  exact h.subset_of_mem (hy hz)
  /-
    🎉 no goals
  -/


theorem isTransitive_iff_sUnion_subset : x.IsTransitive ↔ (⋃₀ x : ZFSet) ⊆ x := by
  /-
    x : ZFSet.{u}
    ⊢ Iff x.IsTransitive (HasSubset.Subset x.sUnion x)
  -/
  constructor <;>
  /-
    case mp
    x : ZFSet.{u}
    ⊢ x.IsTransitive → HasSubset.Subset x.sUnion x
  -/
  intro h y hy
    /-
      case mp
      x : ZFSet.{u}
      h : x.IsTransitive
      y : ZFSet.{u}
      hy : Membership.mem x.sUnion y
      ⊢ Membership.mem x y
    -/
  · obtain ⟨z, hz, hz'⟩ := mem_sUnion.1 hy
    /-
      case mp.intro.intro
      x : ZFSet.{u}
      h : x.IsTransitive
      y : ZFSet.{u}
      hy : Membership.mem x.sUnion y
      z : ZFSet.{u}
      hz : Membership.mem x z
      hz' : Membership.mem z y
      ⊢ Membership.mem x y
    -/
    exact h.mem_trans hz' hz
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : ZFSet.{u}
      h : HasSubset.Subset x.sUnion x
      y : ZFSet.{u}
      hy : Membership.mem x y
      ⊢ HasSubset.Subset y x
    -/
  · exact fun z hz ↦ h <| mem_sUnion_of_mem hz hy
    /-
      🎉 no goals
    -/


alias ⟨IsTransitive.sUnion_subset, _⟩ := isTransitive_iff_sUnion_subset


theorem isTransitive_iff_subset_powerset : x.IsTransitive ↔ x ⊆ powerset x :=
  ⟨fun h _ hy => mem_powerset.2 <| h.subset_of_mem hy, fun H _ hy _ hz => mem_powerset.1 (H hy) hz⟩


alias ⟨IsTransitive.subset_powerset, _⟩ := isTransitive_iff_subset_powerset


/-- A set `x` is a von Neumann ordinal when it's a transitive set, that's transitive under `∈`. We
prove that this further implies that `x` is well-ordered under `∈` in `isOrdinal_iff_isWellOrder`.

The transitivity condition `a ∈ b → b ∈ c → a ∈ c` can be written without assuming `a ∈ x` and
`b ∈ x`. The lemma `isOrdinal_iff_isTrans` shows this condition is equivalent to the usual one. -/
structure IsOrdinal (x : ZFSet) : Prop where
  /-- An ordinal is a transitive set. -/
  isTransitive : x.IsTransitive
  /-- The membership operation within an ordinal is transitive. -/
  mem_trans' {y z w : ZFSet} : y ∈ z → z ∈ w → w ∈ x → y ∈ w


theorem subset_of_mem (h : x.IsOrdinal) : y ∈ x → y ⊆ x :=
  h.isTransitive.subset_of_mem


theorem mem_trans (h : z.IsOrdinal) : x ∈ y → y ∈ z → x ∈ z :=
  h.isTransitive.mem_trans


protected theorem isTrans (h : x.IsOrdinal) : IsTrans x.toSet (Subrel (· ∈ ·) _) :=
  ⟨fun _ _ c hab hbc => h.mem_trans' hab hbc c.2⟩


/-- The simplified form of transitivity used within `IsOrdinal` yields an equivalent definition to
the standard one. -/
theorem _root_.ZFSet.isOrdinal_iff_isTrans :
    x.IsOrdinal ↔ x.IsTransitive ∧ IsTrans x.toSet (Subrel (· ∈ ·) _) where
  mp h := ⟨h.isTransitive, h.isTrans⟩
  mpr := by
    /-
      x : ZFSet.{u}
      ⊢ And x.IsTransitive (IsTrans (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem  …
    -/
    rintro ⟨h₁, ⟨h₂⟩⟩
    /-
      case intro.mk
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : ∀ (a b c : ↑x.toSet), Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet  …
      ⊢ x.IsOrdinal
    -/
    refine ⟨h₁, fun {y z w} hyz hzw hwx ↦ ?_⟩
    /-
      case intro.mk
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : ∀ (a b c : ↑x.toSet), Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet  …
      y z w : ZFSet.{u}
      hyz : Membership.mem z y
      hzw : Membership.mem w z
      hwx : Membership.mem x w
      ⊢ Membership.mem w y
    -/
    have hzx := h₁.mem_trans hzw hwx
    /-
      case intro.mk
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : ∀ (a b c : ↑x.toSet), Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet  …
      y z w : ZFSet.{u}
      hyz : Membership.mem z y
      hzw : Membership.mem w z
      hwx : Membership.mem x w
      hzx : Membership.mem x z
      ⊢ Membership.mem w y
    -/
    exact h₂ ⟨y, h₁.mem_trans hyz hzx⟩ ⟨z, hzx⟩ ⟨w, hwx⟩ hyz hzw
    /-
      🎉 no goals
    -/


protected theorem mem (hx : x.IsOrdinal) (hy : y ∈ x) : y.IsOrdinal :=
  have := hx.isTrans
  let f : Subrel (· ∈ ·) y.toSet ↪r Subrel (· ∈ ·) x.toSet :=
    Subrel.inclusionEmbedding (· ∈ ·) (hx.subset_of_mem hy)
  isOrdinal_iff_isTrans.2 ⟨fun _ hz _ ha ↦ hx.mem_trans' ha hz hy, f.isTrans⟩


/-- An ordinal is a transitive set of transitive sets. -/
theorem _root_.ZFSet.isOrdinal_iff_forall_mem_isTransitive :
    x.IsOrdinal ↔ x.IsTransitive ∧ ∀ y ∈ x, y.IsTransitive where
  mp h := ⟨h.isTransitive, fun _ hy ↦ (h.mem hy).isTransitive⟩
  mpr := fun ⟨h₁, h₂⟩ ↦ ⟨h₁, fun hyz hzw hwx ↦ (h₂ _ hwx).mem_trans hyz hzw⟩


/-- An ordinal is a transitive set of ordinals. -/
theorem _root_.ZFSet.isOrdinal_iff_forall_mem_isOrdinal :
    x.IsOrdinal ↔ x.IsTransitive ∧ ∀ y ∈ x, y.IsOrdinal where
  mp h := ⟨h.isTransitive, fun _ ↦ h.mem⟩
  mpr := fun ⟨h₁, h₂⟩ ↦ isOrdinal_iff_forall_mem_isTransitive.2
    ⟨h₁, fun y hy ↦ (h₂ y hy).isTransitive⟩


theorem subset_iff_eq_or_mem (hx : x.IsOrdinal) (hy : y.IsOrdinal) : x ⊆ y ↔ x = y ∨ x ∈ y := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Iff (HasSubset.Subset x y) (Or (Eq x y) (Membership.mem y x))
  -/
  constructor
    /-
      case mp
      x y : ZFSet.{u}
      hx : x.IsOrdinal
      hy : y.IsOrdinal
      ⊢ HasSubset.Subset x y → Or (Eq x y) (Membership.mem y x)
    -/
  · revert hx hy
    /-
      case mp
      x y : ZFSet.{u}
      ⊢ x.IsOrdinal → y.IsOrdinal → HasSubset.Subset x y → Or (Eq x y) (Membership.m …
    -/
    apply Sym2.GameAdd.induction mem_wf _ x y
    /-
      x y : ZFSet.{u}
      ⊢ ∀ (a₁ b₁ : ZFSet.{u}), (∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Me …
    -/
    intro x y IH hx hy hxy
    /-
      x✝ y✝ x y : ZFSet.{u}
      IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
      hx : x.IsOrdinal
      hy : y.IsOrdinal
      hxy : HasSubset.Subset x y
      ⊢ Or (Eq x y) (Membership.mem y x)
    -/
    by_cases hyx : y ⊆ x
      /-
        case pos
        x✝ y✝ x y : ZFSet.{u}
        IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
        hx : x.IsOrdinal
        hy : y.IsOrdinal
        hxy : HasSubset.Subset x y
        hyx : HasSubset.Subset y x
        ⊢ Or (Eq x y) (Membership.mem y x)
      -/
    · exact Or.inl (subset_antisymm hxy hyx)
      /-
        🎉 no goals
      -/
      /-
        case neg
        x✝ y✝ x y : ZFSet.{u}
        IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
        hx : x.IsOrdinal
        hy : y.IsOrdinal
        hxy : HasSubset.Subset x y
        hyx : Not (HasSubset.Subset y x)
        ⊢ Or (Eq x y) (Membership.mem y x)
      -/
    · obtain ⟨m, hm, hm'⟩ := mem_wf.has_min (y.toSet \ x.toSet) (Set.diff_nonempty.2 hyx)
      /-
        case neg.intro.intro
        x✝ y✝ x y : ZFSet.{u}
        IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
        hx : x.IsOrdinal
        hy : y.IsOrdinal
        hxy : HasSubset.Subset x y
        hyx : Not (HasSubset.Subset y x)
        m : ZFSet.{u}
        hm : Membership.mem (SDiff.sdiff y.toSet x.toSet) m
        hm' : ∀ (x_1 : ZFSet.{u}), Membership.mem (SDiff.sdiff y.toSet x.toSet) x_1 →  …
        ⊢ Or (Eq x y) (Membership.mem y x)
      -/
      have hmy : m ∈ y := show m ∈ y.toSet from Set.mem_of_mem_diff hm
      have hmx : m ⊆ x := by
        intro z hzm
        by_contra hzx
        exact hm' _ ⟨hy.mem_trans hzm hmy, hzx⟩ hzm
      /-
        case neg.intro.intro
        x✝ y✝ x y : ZFSet.{u}
        IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
        hx : x.IsOrdinal
        hy : y.IsOrdinal
        hxy : HasSubset.Subset x y
        hyx : Not (HasSubset.Subset y x)
        m : ZFSet.{u}
        hm : Membership.mem (SDiff.sdiff y.toSet x.toSet) m
        hm' : ∀ (x_1 : ZFSet.{u}), Membership.mem (SDiff.sdiff y.toSet x.toSet) x_1 →  …
        hmy : Membership.mem y m
        hmx : HasSubset.Subset m x
        ⊢ Or (Eq x y) (Membership.mem y x)
      -/
      obtain rfl | H := IH m x (Sym2.GameAdd.fst_snd hmy) (hy.mem hmy) hx hmx
        /-
          case neg.intro.intro.inl
          x y✝ y : ZFSet.{u}
          hy : y.IsOrdinal
          m : ZFSet.{u}
          hmy : Membership.mem y m
          IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
          hx : m.IsOrdinal
          hxy : HasSubset.Subset m y
          hyx : Not (HasSubset.Subset y m)
          hm : Membership.mem (SDiff.sdiff y.toSet m.toSet) m
          hm' : ∀ (x : ZFSet.{u}), Membership.mem (SDiff.sdiff y.toSet m.toSet) x → Not  …
          hmx : HasSubset.Subset m m
          ⊢ Or (Eq m y) (Membership.mem y m)
        -/
      · exact Or.inr hmy
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.intro.inr
          x✝ y✝ x y : ZFSet.{u}
          IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
          hx : x.IsOrdinal
          hy : y.IsOrdinal
          hxy : HasSubset.Subset x y
          hyx : Not (HasSubset.Subset y x)
          m : ZFSet.{u}
          hm : Membership.mem (SDiff.sdiff y.toSet x.toSet) m
          hm' : ∀ (x_1 : ZFSet.{u}), Membership.mem (SDiff.sdiff y.toSet x.toSet) x_1 →  …
          hmy : Membership.mem y m
          hmx : HasSubset.Subset m x
          H : Membership.mem x m
          ⊢ Or (Eq x y) (Membership.mem y x)
        -/
      · cases Set.not_mem_of_mem_diff hm H
        /-
          🎉 no goals
        -/
    /-
      case mpr
      x y : ZFSet.{u}
      hx : x.IsOrdinal
      hy : y.IsOrdinal
      ⊢ Or (Eq x y) (Membership.mem y x) → HasSubset.Subset x y
    -/
  · rintro (rfl | h)
      /-
        case mpr.inl
        x : ZFSet.{u}
        hx hy : x.IsOrdinal
        ⊢ HasSubset.Subset x x
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        x y : ZFSet.{u}
        hx : x.IsOrdinal
        hy : y.IsOrdinal
        h : Membership.mem y x
        ⊢ HasSubset.Subset x y
      -/
    · exact hy.subset_of_mem h
      /-
        🎉 no goals
      -/


alias ⟨eq_or_mem_of_subset, _⟩ := subset_iff_eq_or_mem


theorem mem_of_subset_of_mem (h : x.IsOrdinal) (hz : z.IsOrdinal) (hx : x ⊆ y) (hy : y ∈ z) :
    x ∈ z := by
  /-
    x y z : ZFSet.{u}
    h : x.IsOrdinal
    hz : z.IsOrdinal
    hx : HasSubset.Subset x y
    hy : Membership.mem z y
    ⊢ Membership.mem z x
  -/
  obtain rfl | hx := h.eq_or_mem_of_subset (hz.mem hy) hx
    /-
      case inl
      x z : ZFSet.{u}
      h : x.IsOrdinal
      hz : z.IsOrdinal
      hx : HasSubset.Subset x x
      hy : Membership.mem z x
      ⊢ Membership.mem z x
    -/
  · exact hy
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : ZFSet.{u}
      h : x.IsOrdinal
      hz : z.IsOrdinal
      hx✝ : HasSubset.Subset x y
      hy : Membership.mem z y
      hx : Membership.mem y x
      ⊢ Membership.mem z x
    -/
  · exact hz.mem_trans hx hy
    /-
      🎉 no goals
    -/


theorem not_mem_iff_subset (hx : x.IsOrdinal) (hy : y.IsOrdinal) : x ∉ y ↔ y ⊆ x := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Iff (Not (Membership.mem y x)) (HasSubset.Subset y x)
  -/
  refine ⟨?_, fun hxy hyx ↦ mem_irrefl _ (hxy hyx)⟩
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Not (Membership.mem y x) → HasSubset.Subset y x
  -/
  revert hx hy
  /-
    x y : ZFSet.{u}
    ⊢ x.IsOrdinal → y.IsOrdinal → Not (Membership.mem y x) → HasSubset.Subset y x
  -/
  apply Sym2.GameAdd.induction mem_wf _ x y
  /-
    x y : ZFSet.{u}
    ⊢ ∀ (a₁ b₁ : ZFSet.{u}), (∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Me …
  -/
  intros x y IH hx hy hyx z hzy
  /-
    x✝ y✝ x y : ZFSet.{u}
    IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    hyx : Not (Membership.mem y x)
    z : ZFSet.{u}
    hzy : Membership.mem y z
    ⊢ Membership.mem x z
  -/
  by_contra hzx
  /-
    x✝ y✝ x y : ZFSet.{u}
    IH : ∀ (a₂ b₂ : ZFSet.{u}), Sym2.GameAdd (fun x1 x2 => Membership.mem x2 x1) ( …
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    hyx : Not (Membership.mem y x)
    z : ZFSet.{u}
    hzy : Membership.mem y z
    hzx : Not (Membership.mem x z)
    ⊢ False
  -/
  exact hyx (mem_of_subset_of_mem hx hy (IH z x (Sym2.GameAdd.fst_snd hzy) (hy.mem hzy) hx hzx) hzy)
  /-
    🎉 no goals
  -/


theorem not_subset_iff_mem (hx : x.IsOrdinal) (hy : y.IsOrdinal) : ¬ x ⊆ y ↔ y ∈ x := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Iff (Not (HasSubset.Subset x y)) (Membership.mem x y)
  -/
  rw [not_iff_comm, not_mem_iff_subset hy hx]
  /-
    🎉 no goals
  -/


theorem mem_or_subset (hx : x.IsOrdinal) (hy : y.IsOrdinal) : x ∈ y ∨ y ⊆ x := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Or (Membership.mem y x) (HasSubset.Subset y x)
  -/
  rw [or_iff_not_imp_left, not_mem_iff_subset hx hy]
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ HasSubset.Subset y x → HasSubset.Subset y x
  -/
  exact id
  /-
    🎉 no goals
  -/


theorem subset_total (hx : x.IsOrdinal) (hy : y.IsOrdinal) : x ⊆ y ∨ y ⊆ x := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Or (HasSubset.Subset x y) (HasSubset.Subset y x)
  -/
  obtain h | h := mem_or_subset hx hy
    /-
      case inl
      x y : ZFSet.{u}
      hx : x.IsOrdinal
      hy : y.IsOrdinal
      h : Membership.mem y x
      ⊢ Or (HasSubset.Subset x y) (HasSubset.Subset y x)
    -/
  · exact Or.inl (hy.subset_of_mem h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y : ZFSet.{u}
      hx : x.IsOrdinal
      hy : y.IsOrdinal
      h : HasSubset.Subset y x
      ⊢ Or (HasSubset.Subset x y) (HasSubset.Subset y x)
    -/
  · exact Or.inr h
    /-
      🎉 no goals
    -/


theorem mem_trichotomous (hx : x.IsOrdinal) (hy : y.IsOrdinal) : x ∈ y ∨ x = y ∨ y ∈ x := by
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Or (Membership.mem y x) (Or (Eq x y) (Membership.mem x y))
  -/
  rw [eq_comm, ← subset_iff_eq_or_mem hy hx]
  /-
    x y : ZFSet.{u}
    hx : x.IsOrdinal
    hy : y.IsOrdinal
    ⊢ Or (Membership.mem y x) (HasSubset.Subset y x)
  -/
  exact mem_or_subset hx hy
  /-
    🎉 no goals
  -/


protected theorem isTrichotomous (h : x.IsOrdinal) : IsTrichotomous x.toSet (Subrel (· ∈ ·) _) :=
                            /-
                              x : ZFSet.{u}
                              h : x.IsOrdinal
                              x✝¹ x✝ : ↑x.toSet
                              a : ZFSet.{u}
                              ha : Membership.mem x.toSet a
                              b : ZFSet.{u}
                              hb : Membership.mem x.toSet b
                              ⊢ Or (Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet ⟨a, ha⟩ ⟨b, hb⟩) (Or  …
                            -/
  ⟨fun ⟨a, ha⟩ ⟨b, hb⟩ ↦ by simpa using mem_trichotomous (h.mem ha) (h.mem hb)⟩
                            /-
                              🎉 no goals
                            -/


/-- An ordinal is a transitive set, trichotomous under membership. -/
theorem _root_.ZFSet.isOrdinal_iff_isTrichotomous :
    x.IsOrdinal ↔ x.IsTransitive ∧ IsTrichotomous x.toSet (Subrel (· ∈ ·) _) where
  mp h := ⟨h.isTransitive, h.isTrichotomous⟩
  mpr := by
    /-
      x : ZFSet.{u}
      ⊢ And x.IsTransitive (IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membersh …
    -/
    rintro ⟨h₁, h₂⟩
    /-
      case intro
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
      ⊢ x.IsOrdinal
    -/
    rw [isOrdinal_iff_isTrans]
    /-
      case intro
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
      ⊢ And x.IsTransitive (IsTrans (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem  …
    -/
    refine ⟨h₁, ⟨@fun y z w hyz hzw ↦ ?_⟩⟩
    /-
      case intro
      x : ZFSet.{u}
      h₁ : x.IsTransitive
      h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
      y z w : ↑x.toSet
      hyz : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y z
      hzw : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet z w
      ⊢ Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y w
    -/
    obtain hyw | rfl | hwy := trichotomous_of (Subrel (· ∈ ·) _) y w
      /-
        case intro.inl
        x : ZFSet.{u}
        h₁ : x.IsTransitive
        h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
        y z w : ↑x.toSet
        hyz : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y z
        hzw : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet z w
        hyw : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y w
        ⊢ Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y w
      -/
    · exact hyw
      /-
        🎉 no goals
      -/
      /-
        case intro.inr.inl
        x : ZFSet.{u}
        h₁ : x.IsTransitive
        h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
        y z : ↑x.toSet
        hyz : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y z
        hzw : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet z y
        ⊢ Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y y
      -/
    · cases asymm hyz hzw
      /-
        🎉 no goals
      -/
      /-
        case intro.inr.inr
        x : ZFSet.{u}
        h₁ : x.IsTransitive
        h₂ : IsTrichotomous (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.t …
        y z w : ↑x.toSet
        hyz : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y z
        hzw : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet z w
        hwy : Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet w y
        ⊢ Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet y w
      -/
    · cases mem_wf.asymmetric₃ _ _ _ hyz hzw hwy
      /-
        🎉 no goals
      -/


protected theorem isWellOrder (h : x.IsOrdinal) : IsWellOrder x.toSet (Subrel (· ∈ ·) _) where
  wf := (Subrel.relEmbedding _ _).wellFounded mem_wf
  trans := h.isTrans.1
  trichotomous := h.isTrichotomous.1


/-- An ordinal is a transitive set, well-ordered under membership. -/
theorem _root_.ZFSet.isOrdinal_iff_isWellOrder : x.IsOrdinal ↔
    x.IsTransitive ∧ IsWellOrder x.toSet (Subrel (· ∈ ·) _) := by
  /-
    x : ZFSet.{u}
    ⊢ Iff x.IsOrdinal (And x.IsTransitive (IsWellOrder (↑x.toSet) (Subrel (fun x1  …
  -/
  use fun h ↦ ⟨h.isTransitive, h.isWellOrder⟩
  /-
    case mpr
    x : ZFSet.{u}
    ⊢ And x.IsTransitive (IsWellOrder (↑x.toSet) (Subrel (fun x1 x2 => Membership. …
  -/
  rintro ⟨h₁, h₂⟩
  /-
    case mpr.intro
    x : ZFSet.{u}
    h₁ : x.IsTransitive
    h₂ : IsWellOrder (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet)
    ⊢ x.IsOrdinal
  -/
  refine isOrdinal_iff_isTrans.2 ⟨h₁, ?_⟩
  /-
    case mpr.intro
    x : ZFSet.{u}
    h₁ : x.IsTransitive
    h₂ : IsWellOrder (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet)
    ⊢ IsTrans (↑x.toSet) (Subrel (fun x1 x2 => Membership.mem x2 x1) x.toSet)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem isOrdinal_empty : IsOrdinal ∅ :=
  ⟨isTransitive_empty, fun _ _ H ↦ (not_mem_empty _ H).elim⟩


/-- The **Burali-Forti paradox**: ordinals form a proper class. -/
theorem isOrdinal_not_mem_univ : IsOrdinal ∉ Class.univ.{u} := by
  /-
    ⊢ Not (Membership.mem Class.univ ZFSet.IsOrdinal)
  -/
  rintro ⟨x, hx, -⟩
  suffices IsOrdinal x by
    apply Class.mem_irrefl x
    rwa [Class.coe_mem, hx]
  /-
    case intro.intro
    x : ZFSet.{u}
    hx : Eq (↑x) ZFSet.IsOrdinal
    ⊢ x.IsOrdinal
  -/
  refine ⟨fun y hy z hz ↦ ?_, fun hyz hzw hwx ↦ ?_⟩ <;> rw [← Class.coe_apply, hx] at *
  /-
    case intro.intro.refine_1
    x : ZFSet.{u}
    hx : Eq ZFSet.IsOrdinal ZFSet.IsOrdinal
    y : ZFSet.{u}
    hy : y.IsOrdinal
    z : ZFSet.{u}
    hz : ↑y z
    ⊢ z.IsOrdinal
  -/
  exacts [hy.mem hz, hwx.mem_trans hyz hzw]
  /-
    🎉 no goals
  -/


