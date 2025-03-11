/-- Finitely supported product of finsets. -/
def dfinsupp (s : Finset ι) (t : ∀ i, Finset (α i)) : Finset (Π₀ i, α i) :=
  (s.pi t).map
    ⟨fun f => DFinsupp.mk s fun i => f i i.2, by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → Zero (α i)
        s✝ : Finset ι
        f : DFinsupp fun i => α i
        t✝ : (i : ι) → Finset (α i)
        s : Finset ι
        t : (i : ι) → Finset (α i)
        ⊢ Function.Injective fun f => DFinsupp.mk s fun i => f ↑i ⋯
      -/
      refine (mk_injective _).comp fun f g h => ?_
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → Zero (α i)
        s✝ : Finset ι
        f✝ : DFinsupp fun i => α i
        t✝ : (i : ι) → Finset (α i)
        s : Finset ι
        t : (i : ι) → Finset (α i)
        f g : (a : ι) → Membership.mem s a → α a
        h : Eq (fun i => f ↑i ⋯) fun i => g ↑i ⋯
        ⊢ Eq f g
      -/
      ext i hi
      /-
        case h.h
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → Zero (α i)
        s✝ : Finset ι
        f✝ : DFinsupp fun i => α i
        t✝ : (i : ι) → Finset (α i)
        s : Finset ι
        t : (i : ι) → Finset (α i)
        f g : (a : ι) → Membership.mem s a → α a
        h : Eq (fun i => f ↑i ⋯) fun i => g ↑i ⋯
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (f i hi) (g i hi)
      -/
      convert congr_fun h ⟨i, hi⟩⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem card_dfinsupp (s : Finset ι) (t : ∀ i, Finset (α i)) : #(s.dfinsupp t) = ∏ i ∈ s, #(t i) :=
  (card_map _).trans <| card_pi _ _


theorem mem_dfinsupp_iff : f ∈ s.dfinsupp t ↔ f.support ⊆ s ∧ ∀ i ∈ s, f i ∈ t i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (α i)
    s : Finset ι
    f : DFinsupp fun i => α i
    t : (i : ι) → Finset (α i)
    inst✝ : (i : ι) → DecidableEq (α i)
    ⊢ Iff (Membership.mem (s.dfinsupp t) f) (And (HasSubset.Subset f.support s) (∀ …
  -/
  refine mem_map.trans ⟨?_, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      ⊢ (Exists fun a => And (Membership.mem (s.pi t) a) (Eq ({ toFun := fun f => DF …
    -/
  · rintro ⟨f, hf, rfl⟩
    /-
      case refine_1.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      f : (a : ι) → Membership.mem s a → α a
      hf : Membership.mem (s.pi t) f
      ⊢ And (HasSubset.Subset ({ toFun := fun f => DFinsupp.mk s fun i => f ↑i ⋯, in …
    -/
    rw [Function.Embedding.coeFn_mk] -- Porting note: added to avoid heartbeat timeout
    /-
      case refine_1.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      f : (a : ι) → Membership.mem s a → α a
      hf : Membership.mem (s.pi t) f
      ⊢ And (HasSubset.Subset (DFinsupp.mk s fun i => f ↑i ⋯).support s) (∀ (i : ι), …
    -/
    refine ⟨support_mk_subset, fun i hi => ?_⟩
    /-
      case refine_1.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      f : (a : ι) → Membership.mem s a → α a
      hf : Membership.mem (s.pi t) f
      i : ι
      hi : Membership.mem s i
      ⊢ Membership.mem (t i) ((DFinsupp.mk s fun i => f ↑i ⋯) i)
    -/
    convert mem_pi.1 hf i hi
    /-
      case h.e'_5
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      f : (a : ι) → Membership.mem s a → α a
      hf : Membership.mem (s.pi t) f
      i : ι
      hi : Membership.mem s i
      ⊢ Eq ((DFinsupp.mk s fun i => f ↑i ⋯) i) (f i hi)
    -/
    exact mk_of_mem hi
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      ⊢ And (HasSubset.Subset f.support s) (∀ (i : ι), Membership.mem s i → Membersh …
    -/
  · refine fun h => ⟨fun i _ => f i, mem_pi.2 h.2, ?_⟩
    /-
      case refine_2
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      h : And (HasSubset.Subset f.support s) (∀ (i : ι), Membership.mem s i → Member …
      ⊢ Eq ({ toFun := fun f => DFinsupp.mk s fun i => f ↑i ⋯, inj' := ⋯ } fun i x = …
    -/
    ext i
    /-
      case refine_2.h
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      h : And (HasSubset.Subset f.support s) (∀ (i : ι), Membership.mem s i → Member …
      i : ι
      ⊢ Eq (({ toFun := fun f => DFinsupp.mk s fun i => f ↑i ⋯, inj' := ⋯ } fun i x  …
    -/
    dsimp
    /-
      case refine_2.h
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      t : (i : ι) → Finset (α i)
      inst✝ : (i : ι) → DecidableEq (α i)
      h : And (HasSubset.Subset f.support s) (∀ (i : ι), Membership.mem s i → Member …
      i : ι
      ⊢ Eq (ite (Membership.mem s i) (f i) 0) (f i)
    -/
    exact ite_eq_left_iff.2 fun hi => (not_mem_support_iff.1 fun H => hi <| h.1 H).symm
    /-
      🎉 no goals
    -/


/-- When `t` is supported on `s`, `f ∈ s.dfinsupp t` precisely means that `f` is pointwise in `t`.
-/
@[simp]
theorem mem_dfinsupp_iff_of_support_subset {t : Π₀ i, Finset (α i)} (ht : t.support ⊆ s) :
    f ∈ s.dfinsupp t ↔ ∀ i, f i ∈ t i := by
  refine mem_dfinsupp_iff.trans (forall_and.symm.trans <| forall_congr' fun i =>
      ⟨ fun h => ?_,
        fun h => ⟨fun hi => ht <| mem_support_iff.2 fun H => mem_support_iff.1 hi ?_, fun _ => h⟩⟩)
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      inst✝ : (i : ι) → DecidableEq (α i)
      t : DFinsupp fun i => Finset (α i)
      ht : HasSubset.Subset t.support s
      i : ι
      h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
      ⊢ Membership.mem (t i) (f i)
    -/
  · by_cases hi : i ∈ s
      /-
        case pos
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (α i)
        s : Finset ι
        f : DFinsupp fun i => α i
        inst✝ : (i : ι) → DecidableEq (α i)
        t : DFinsupp fun i => Finset (α i)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Membership.mem s i
        ⊢ Membership.mem (t i) (f i)
      -/
    · exact h.2 hi
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (α i)
        s : Finset ι
        f : DFinsupp fun i => α i
        inst✝ : (i : ι) → DecidableEq (α i)
        t : DFinsupp fun i => Finset (α i)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Not (Membership.mem s i)
        ⊢ Membership.mem (t i) (f i)
      -/
    · rw [not_mem_support_iff.1 (mt h.1 hi), not_mem_support_iff.1 (not_mem_mono ht hi)]
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (α i)
        s : Finset ι
        f : DFinsupp fun i => α i
        inst✝ : (i : ι) → DecidableEq (α i)
        t : DFinsupp fun i => Finset (α i)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Not (Membership.mem s i)
        ⊢ Membership.mem 0 0
      -/
      exact zero_mem_zero
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (α i)
      s : Finset ι
      f : DFinsupp fun i => α i
      inst✝ : (i : ι) → DecidableEq (α i)
      t : DFinsupp fun i => Finset (α i)
      ht : HasSubset.Subset t.support s
      i : ι
      h : Membership.mem (t i) (f i)
      hi : Membership.mem f.support i
      H : Eq (t i) 0
      ⊢ Eq (f i) 0
    -/
  · rwa [H, mem_zero] at h
    /-
      🎉 no goals
    -/


/-- Pointwise `Finset.singleton` bundled as a `DFinsupp`. -/
def singleton (f : Π₀ i, α i) : Π₀ i, Finset (α i) where
  toFun i := {f i}
  support' := f.support'.map fun s => ⟨s.1, fun i => (s.prop i).imp id (congr_arg _)⟩


theorem mem_singleton_apply_iff : a ∈ f.singleton i ↔ a = f i :=
  mem_singleton


/-- Pointwise `Finset.Icc` bundled as a `DFinsupp`. -/
def rangeIcc (f g : Π₀ i, α i) : Π₀ i, Finset (α i) where
  toFun i := Icc (f i) (g i)
  support' := f.support'.bind fun fs => g.support'.map fun gs =>
    ⟨ fs.1 + gs.1,
      fun i => or_iff_not_imp_left.2 fun h => by
        have hf : f i = 0 := (fs.prop i).resolve_left
            (Multiset.not_mem_mono (Multiset.Le.subset <| Multiset.le_add_right _ _) h)
        have hg : g i = 0 := (gs.prop i).resolve_left
            (Multiset.not_mem_mono (Multiset.Le.subset <| Multiset.le_add_left _ _) h)
        /-
          ι : Type u_1
          α : ι → Type u_2
          inst✝² : (i : ι) → Zero (α i)
          inst✝¹ : (i : ι) → PartialOrder (α i)
          inst✝ : (i : ι) → LocallyFiniteOrder (α i)
          f✝ g✝ : DFinsupp fun i => α i
          i✝ : ι
          a : α i✝
          f g : DFinsupp fun i => α i
          fs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
          gs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g.toFun i) 0)
          i : ι
          h : Not (Membership.mem (HAdd.hAdd ↑fs ↑gs) i)
          hf : Eq (f i) 0
          hg : Eq (g i) 0
          ⊢ Eq ((fun i => Finset.Icc (f i) (g i)) i) 0
        -/
        simp_rw [hf, hg]
        /-
          ι : Type u_1
          α : ι → Type u_2
          inst✝² : (i : ι) → Zero (α i)
          inst✝¹ : (i : ι) → PartialOrder (α i)
          inst✝ : (i : ι) → LocallyFiniteOrder (α i)
          f✝ g✝ : DFinsupp fun i => α i
          i✝ : ι
          a : α i✝
          f g : DFinsupp fun i => α i
          fs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
          gs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g.toFun i) 0)
          i : ι
          h : Not (Membership.mem (HAdd.hAdd ↑fs ↑gs) i)
          hf : Eq (f i) 0
          hg : Eq (g i) 0
          ⊢ Eq (Finset.Icc 0 0) 0
        -/
        exact Icc_self _⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem rangeIcc_apply (f g : Π₀ i, α i) (i : ι) : f.rangeIcc g i = Icc (f i) (g i) := rfl


theorem mem_rangeIcc_apply_iff : a ∈ f.rangeIcc g i ↔ f i ≤ a ∧ a ≤ g i := mem_Icc


theorem support_rangeIcc_subset [DecidableEq ι] [∀ i, DecidableEq (α i)] :
    (f.rangeIcc g).support ⊆ f.support ∪ g.support := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : (i : ι) → Zero (α i)
    inst✝³ : (i : ι) → PartialOrder (α i)
    inst✝² : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    ⊢ HasSubset.Subset (f.rangeIcc g).support (Union.union f.support g.support)
  -/
  refine fun x hx => ?_
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : (i : ι) → Zero (α i)
    inst✝³ : (i : ι) → PartialOrder (α i)
    inst✝² : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    x : ι
    hx : Membership.mem (f.rangeIcc g).support x
    ⊢ Membership.mem (Union.union f.support g.support) x
  -/
  by_contra h
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : (i : ι) → Zero (α i)
    inst✝³ : (i : ι) → PartialOrder (α i)
    inst✝² : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    x : ι
    hx : Membership.mem (f.rangeIcc g).support x
    h : Not (Membership.mem (Union.union f.support g.support) x)
    ⊢ False
  -/
  refine not_mem_support_iff.2 ?_ hx
  rw [rangeIcc_apply, not_mem_support_iff.1 (not_mem_mono subset_union_left h),
    not_mem_support_iff.1 (not_mem_mono subset_union_right h)]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : (i : ι) → Zero (α i)
    inst✝³ : (i : ι) → PartialOrder (α i)
    inst✝² : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    x : ι
    hx : Membership.mem (f.rangeIcc g).support x
    h : Not (Membership.mem (Union.union f.support g.support) x)
    ⊢ Eq (Finset.Icc 0 0) 0
  -/
  exact Icc_self _
  /-
    🎉 no goals
  -/


/-- Given a finitely supported function `f : Π₀ i, Finset (α i)`, one can define the finset
`f.pi` of all finitely supported functions whose value at `i` is in `f i` for all `i`. -/
def pi (f : Π₀ i, Finset (α i)) : Finset (Π₀ i, α i) := f.support.dfinsupp f


@[simp]
theorem mem_pi {f : Π₀ i, Finset (α i)} {g : Π₀ i, α i} : g ∈ f.pi ↔ ∀ i, g i ∈ f i :=
  mem_dfinsupp_iff_of_support_subset <| Subset.refl _


@[simp]
theorem card_pi (f : Π₀ i, Finset (α i)) : #f.pi = f.prod fun i ↦ #(f i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    f : DFinsupp fun i => Finset (α i)
    ⊢ Eq f.pi.card (f.prod fun i => ↑(f i).card)
  -/
  rw [pi, card_dfinsupp]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    f : DFinsupp fun i => Finset (α i)
    ⊢ Eq (f.support.prod fun i => (f i).card) (f.prod fun i => ↑(f i).card)
  -/
  exact Finset.prod_congr rfl fun i _ => by simp only [Pi.natCast_apply, Nat.cast_id]
  /-
    🎉 no goals
  -/


instance instLocallyFiniteOrder : LocallyFiniteOrder (Π₀ i, α i) :=
  LocallyFiniteOrder.ofIcc (Π₀ i, α i)
    (fun f g => (f.support ∪ g.support).dfinsupp <| f.rangeIcc g)
    (fun f g x => by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : DecidableEq ι
        inst✝³ : (i : ι) → DecidableEq (α i)
        inst✝² : (i : ι) → PartialOrder (α i)
        inst✝¹ : (i : ι) → Zero (α i)
        inst✝ : (i : ι) → LocallyFiniteOrder (α i)
        f g x : DFinsupp fun i => α i
        ⊢ Iff (Membership.mem ((fun f g => (Union.union f.support g.support).dfinsupp  …
      -/
      refine (mem_dfinsupp_iff_of_support_subset <| support_rangeIcc_subset).trans ?_
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : DecidableEq ι
        inst✝³ : (i : ι) → DecidableEq (α i)
        inst✝² : (i : ι) → PartialOrder (α i)
        inst✝¹ : (i : ι) → Zero (α i)
        inst✝ : (i : ι) → LocallyFiniteOrder (α i)
        f g x : DFinsupp fun i => α i
        ⊢ Iff (∀ (i : ι), Membership.mem ((f.rangeIcc g) i) (x i)) (And (LE.le f x) (L …
      -/
      simp_rw [mem_rangeIcc_apply_iff, forall_and]
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : DecidableEq ι
        inst✝³ : (i : ι) → DecidableEq (α i)
        inst✝² : (i : ι) → PartialOrder (α i)
        inst✝¹ : (i : ι) → Zero (α i)
        inst✝ : (i : ι) → LocallyFiniteOrder (α i)
        f g x : DFinsupp fun i => α i
        ⊢ Iff (And (∀ (x_1 : ι), LE.le (f x_1) (x x_1)) (∀ (x_1 : ι), LE.le (x x_1) (g …
      -/
      rfl)
      /-
        🎉 no goals
      -/


theorem Icc_eq : Icc f g = (f.support ∪ g.support).dfinsupp (f.rangeIcc g) := rfl


lemma card_Icc : #(Icc f g) = ∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i)) :=
  card_dfinsupp _ _


lemma card_Ico : #(Ico f g) = (∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i))) - 1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → DecidableEq (α i)
    inst✝² : (i : ι) → PartialOrder (α i)
    inst✝¹ : (i : ι) → Zero (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    ⊢ Eq (Finset.Ico f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ico_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


lemma card_Ioc : #(Ioc f g) = (∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i))) - 1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → DecidableEq (α i)
    inst✝² : (i : ι) → PartialOrder (α i)
    inst✝¹ : (i : ι) → Zero (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    ⊢ Eq (Finset.Ioc f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ioc_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


lemma card_Ioo : #(Ioo f g) = (∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i))) - 2 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → DecidableEq (α i)
    inst✝² : (i : ι) → PartialOrder (α i)
    inst✝¹ : (i : ι) → Zero (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    ⊢ Eq (Finset.Ioo f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ioo_eq_card_Icc_sub_two, card_Icc]
  /-
    🎉 no goals
  -/


lemma card_uIcc : #(uIcc f g) = ∏ i ∈ f.support ∪ g.support, #(uIcc (f i) (g i)) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → DecidableEq (α i)
    inst✝² : (i : ι) → Lattice (α i)
    inst✝¹ : (i : ι) → Zero (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    f g : DFinsupp fun i => α i
    ⊢ Eq (Finset.uIcc f g).card ((Union.union f.support g.support).prod fun i => ( …
  -/
  rw [← support_inf_union_support_sup]; exact card_Icc _ _
                                        /-
                                          🎉 no goals
                                        -/


lemma card_Iic : #(Iic f) = ∏ i ∈ f.support, #(Iic (f i)) := by
  simp_rw [Iic_eq_Icc, card_Icc, DFinsupp.bot_eq_zero, support_zero, empty_union, zero_apply,
    bot_eq_zero]


lemma card_Iio : #(Iio f) = (∏ i ∈ f.support, #(Iic (f i))) - 1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → DecidableEq (α i)
    inst✝¹ : (i : ι) → CanonicallyOrderedAddCommMonoid (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    f : DFinsupp fun i => α i
    ⊢ Eq (Finset.Iio f).card (HSub.hSub (f.support.prod fun i => (Finset.Iic (f i) …
  -/
  rw [card_Iio_eq_card_Iic_sub_one, card_Iic]
  /-
    🎉 no goals
  -/


