@[to_additive]
theorem dfinsupp_prod_mem [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [CommMonoid γ] {S : Type*} [SetLike S γ] [SubmonoidClass S γ]
    (s : S) (f : Π₀ i, β i) (g : ∀ i, β i → γ)
    (h : ∀ c, f c ≠ 0 → g c (f c) ∈ s) : f.prod g ∈ s :=
  prod_mem fun _ hi => h _ <| mem_support_iff.1 hi


theorem dfinsupp_sumAddHom_mem [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] {S : Type*}
    [SetLike S γ] [AddSubmonoidClass S γ] (s : S) (f : Π₀ i, β i) (g : ∀ i, β i →+ γ)
    (h : ∀ c, f c ≠ 0 → g c (f c) ∈ s) : DFinsupp.sumAddHom g f ∈ s := by
  classical
    rw [DFinsupp.sumAddHom_apply]
    exact dfinsupp_sum_mem s f (g ·) h


/-- The supremum of a family of commutative additive submonoids is equal to the range of
`DFinsupp.sumAddHom`; that is, every element in the `iSup` can be produced from taking a finite
number of non-zero elements of `S i`, coercing them to `γ`, and summing them. -/
theorem AddSubmonoid.iSup_eq_mrange_dfinsupp_sumAddHom
    [AddCommMonoid γ] (S : ι → AddSubmonoid γ) :
    iSup S = AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S i).subtype) := by
  /-
    ι : Type u
    γ : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid γ
    S : ι → AddSubmonoid γ
    ⊢ Eq (iSup S) (AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S i).subtype))
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝¹ : DecidableEq ι
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      ⊢ LE.le (iSup S) (AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S i).subty …
    -/
  · apply iSup_le _
    /-
      ι : Type u
      γ : Type w
      inst✝¹ : DecidableEq ι
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      ⊢ ∀ (i : ι), LE.le (S i) (AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S  …
    -/
    intro i y hy
    /-
      ι : Type u
      γ : Type w
      inst✝¹ : DecidableEq ι
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      i : ι
      y : γ
      hy : Membership.mem (S i) y
      ⊢ Membership.mem (AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S i).subty …
    -/
    exact ⟨DFinsupp.single i ⟨y, hy⟩, DFinsupp.sumAddHom_single _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝¹ : DecidableEq ι
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      ⊢ LE.le (AddMonoidHom.mrange (DFinsupp.sumAddHom fun i => (S i).subtype)) (iSu …
    -/
  · rintro x ⟨v, rfl⟩
    /-
      case a.intro
      ι : Type u
      γ : Type w
      inst✝¹ : DecidableEq ι
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      ⊢ Membership.mem (iSup S) ((DFinsupp.sumAddHom fun i => (S i).subtype) v)
    -/
    exact dfinsupp_sumAddHom_mem _ v _ fun i _ => (le_iSup S i : S i ≤ _) (v i).prop
    /-
      🎉 no goals
    -/


/-- The bounded supremum of a family of commutative additive submonoids is equal to the range of
`DFinsupp.sumAddHom` composed with `DFinsupp.filterAddMonoidHom`; that is, every element in the
bounded `iSup` can be produced from taking a finite number of non-zero elements from the `S i` that
satisfy `p i`, coercing them to `γ`, and summing them. -/
theorem AddSubmonoid.bsupr_eq_mrange_dfinsupp_sumAddHom (p : ι → Prop) [DecidablePred p]
    [AddCommMonoid γ] (S : ι → AddSubmonoid γ) :
    ⨆ (i) (_ : p i), S i =
      AddMonoidHom.mrange ((sumAddHom fun i => (S i).subtype).comp (filterAddMonoidHom _ p)) := by
  /-
    ι : Type u
    γ : Type w
    inst✝² : DecidableEq ι
    p : ι → Prop
    inst✝¹ : DecidablePred p
    inst✝ : AddCommMonoid γ
    S : ι → AddSubmonoid γ
    ⊢ Eq (iSup fun i => iSup fun x => S i) (AddMonoidHom.mrange ((DFinsupp.sumAddH …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      ⊢ LE.le (iSup fun i => iSup fun x => S i) (AddMonoidHom.mrange ((DFinsupp.sumA …
    -/
  · refine iSup₂_le fun i hi y hy => ⟨DFinsupp.single i ⟨y, hy⟩, ?_⟩
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      i : ι
      hi : p i
      y : γ
      hy : Membership.mem (S i) y
      ⊢ Eq (((DFinsupp.sumAddHom fun i => (S i).subtype).comp (DFinsupp.filterAddMon …
    -/
    rw [AddMonoidHom.comp_apply, filterAddMonoidHom_apply, filter_single_pos _ _ hi]
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      i : ι
      hi : p i
      y : γ
      hy : Membership.mem (S i) y
      ⊢ Eq ((DFinsupp.sumAddHom fun i => (S i).subtype) (DFinsupp.single i ⟨y, hy⟩)) y
    -/
    exact sumAddHom_single _ _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      ⊢ LE.le (AddMonoidHom.mrange ((DFinsupp.sumAddHom fun i => (S i).subtype).comp …
    -/
  · rintro x ⟨v, rfl⟩
    /-
      case a.intro
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      ⊢ Membership.mem (iSup fun i => iSup fun x => S i) (((DFinsupp.sumAddHom fun i …
    -/
    refine dfinsupp_sumAddHom_mem _ _ _ fun i _ => ?_
    /-
      case a.intro
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      i : ι
      x✝ : Ne (((DFinsupp.filterAddMonoidHom (fun i => Subtype fun x => Membership.m …
      ⊢ Membership.mem (iSup fun i => iSup fun x => S i) ((S i).subtype (((DFinsupp. …
    -/
    refine AddSubmonoid.mem_iSup_of_mem i ?_
    /-
      case a.intro
      ι : Type u
      γ : Type w
      inst✝² : DecidableEq ι
      p : ι → Prop
      inst✝¹ : DecidablePred p
      inst✝ : AddCommMonoid γ
      S : ι → AddSubmonoid γ
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      i : ι
      x✝ : Ne (((DFinsupp.filterAddMonoidHom (fun i => Subtype fun x => Membership.m …
      ⊢ Membership.mem (iSup fun x => S i) ((S i).subtype (((DFinsupp.filterAddMonoi …
    -/
    by_cases hp : p i
      /-
        case pos
        ι : Type u
        γ : Type w
        inst✝² : DecidableEq ι
        p : ι → Prop
        inst✝¹ : DecidablePred p
        inst✝ : AddCommMonoid γ
        S : ι → AddSubmonoid γ
        v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
        i : ι
        x✝ : Ne (((DFinsupp.filterAddMonoidHom (fun i => Subtype fun x => Membership.m …
        hp : p i
        ⊢ Membership.mem (iSup fun x => S i) ((S i).subtype (((DFinsupp.filterAddMonoi …
      -/
    · simp [hp]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        γ : Type w
        inst✝² : DecidableEq ι
        p : ι → Prop
        inst✝¹ : DecidablePred p
        inst✝ : AddCommMonoid γ
        S : ι → AddSubmonoid γ
        v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
        i : ι
        x✝ : Ne (((DFinsupp.filterAddMonoidHom (fun i => Subtype fun x => Membership.m …
        hp : Not (p i)
        ⊢ Membership.mem (iSup fun x => S i) ((S i).subtype (((DFinsupp.filterAddMonoi …
      -/
    · simp [hp]
      /-
        🎉 no goals
      -/


theorem AddSubmonoid.mem_iSup_iff_exists_dfinsupp [AddCommMonoid γ] (S : ι → AddSubmonoid γ)
    (x : γ) : x ∈ iSup S ↔ ∃ f : Π₀ i, S i, DFinsupp.sumAddHom (fun i => (S i).subtype) f = x :=
  SetLike.ext_iff.mp (AddSubmonoid.iSup_eq_mrange_dfinsupp_sumAddHom S) x


/-- A variant of `AddSubmonoid.mem_iSup_iff_exists_dfinsupp` with the RHS fully unfolded. -/
theorem AddSubmonoid.mem_iSup_iff_exists_dfinsupp' [AddCommMonoid γ] (S : ι → AddSubmonoid γ)
    [∀ (i) (x : S i), Decidable (x ≠ 0)] (x : γ) :
    x ∈ iSup S ↔ ∃ f : Π₀ i, S i, (f.sum fun _ xi => ↑xi) = x := by
  /-
    ι : Type u
    γ : Type w
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid γ
    S : ι → AddSubmonoid γ
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (S i) x) → Decidable (N …
    x : γ
    ⊢ Iff (Membership.mem (iSup S) x) (Exists fun f => Eq (f.sum fun x xi => ↑xi) x)
  -/
  rw [AddSubmonoid.mem_iSup_iff_exists_dfinsupp]
  /-
    ι : Type u
    γ : Type w
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid γ
    S : ι → AddSubmonoid γ
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (S i) x) → Decidable (N …
    x : γ
    ⊢ Iff (Exists fun f => Eq ((DFinsupp.sumAddHom fun i => (S i).subtype) f) x) ( …
  -/
  simp_rw [sumAddHom_apply]
  /-
    ι : Type u
    γ : Type w
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid γ
    S : ι → AddSubmonoid γ
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (S i) x) → Decidable (N …
    x : γ
    ⊢ Iff (Exists fun f => Eq (f.sum fun x => ⇑(S x).subtype) x) (Exists fun f =>  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem AddSubmonoid.mem_bsupr_iff_exists_dfinsupp (p : ι → Prop) [DecidablePred p]
    [AddCommMonoid γ] (S : ι → AddSubmonoid γ) (x : γ) :
    (x ∈ ⨆ (i) (_ : p i), S i) ↔
      ∃ f : Π₀ i, S i, DFinsupp.sumAddHom (fun i => (S i).subtype) (f.filter p) = x :=
  SetLike.ext_iff.mp (AddSubmonoid.bsupr_eq_mrange_dfinsupp_sumAddHom p S) x

