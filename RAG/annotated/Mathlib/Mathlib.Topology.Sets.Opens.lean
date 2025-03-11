/-- The type of open subsets of a topological space. -/
structure Opens where
  /-- The underlying set of a bundled `TopologicalSpace.Opens` object. -/
  carrier : Set α
  /-- The `TopologicalSpace.Opens.carrier _` is an open set. -/
  is_open' : IsOpen carrier


instance : SetLike (Opens α) α where
  coe := Opens.carrier
                                              /-
                                                ι : Type u_1
                                                α : Type u_2
                                                β : Type u_3
                                                γ : Type u_4
                                                inst✝² : TopologicalSpace α
                                                inst✝¹ : TopologicalSpace β
                                                inst✝ : TopologicalSpace γ
                                                x✝² x✝¹ : TopologicalSpace.Opens α
                                                carrier✝¹ : Set α
                                                is_open'✝¹ : IsOpen carrier✝¹
                                                carrier✝ : Set α
                                                is_open'✝ : IsOpen carrier✝
                                                x✝ : Eq { carrier := carrier✝¹, is_open' := is_open'✝¹ }.carrier { carrier :=  …
                                                ⊢ Eq { carrier := carrier✝¹, is_open' := is_open'✝¹ } { carrier := carrier✝, i …
                                              -/
  coe_injective' := fun ⟨_, _⟩ ⟨_, _⟩ _ => by congr
                                              /-
                                                🎉 no goals
                                              -/


instance : CanLift (Set α) (Opens α) (↑) IsOpen :=
  ⟨fun s h => ⟨⟨s, h⟩, rfl⟩⟩


instance instSecondCountableOpens [SecondCountableTopology α] (U : Opens α) :
    SecondCountableTopology U := inferInstanceAs (SecondCountableTopology U.1)


theorem «forall» {p : Opens α → Prop} : (∀ U, p U) ↔ ∀ (U : Set α) (hU : IsOpen U), p ⟨U, hU⟩ :=
  ⟨fun h _ _ => h _, fun h _ => h _ _⟩


@[simp] theorem carrier_eq_coe (U : Opens α) : U.1 = ↑U := rfl


/-- the coercion `Opens α → Set α` applied to a pair is the same as taking the first component -/
@[simp]
theorem coe_mk {U : Set α} {hU : IsOpen U} : ↑(⟨U, hU⟩ : Opens α) = U :=
  rfl


@[simp]
theorem mem_mk {x : α} {U : Set α} {h : IsOpen U} : x ∈ mk U h ↔ x ∈ U := Iff.rfl

-- Porting note: removed @[simp] because LHS simplifies to `∃ x, x ∈ U`

protected theorem nonempty_coeSort {U : Opens α} : Nonempty U ↔ (U : Set α).Nonempty :=
  Set.nonempty_coe_sort

-- TODO: should this theorem be proved for a `SetLike`?

protected theorem nonempty_coe {U : Opens α} : (U : Set α).Nonempty ↔ ∃ x, x ∈ U :=
  Iff.rfl


@[ext] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: replace with `∀ x, x ∈ U ↔ x ∈ V`
theorem ext {U V : Opens α} (h : (U : Set α) = V) : U = V :=
  SetLike.coe_injective h

-- Porting note: removed @[simp], simp can prove it

theorem coe_inj {U V : Opens α} : (U : Set α) = V ↔ U = V :=
  SetLike.ext'_iff.symm


/-- A version of `Set.inclusion` not requiring definitional abuse -/
abbrev inclusion {U V : Opens α} (h : U ≤ V) : U → V := Set.inclusion h


protected theorem isOpen (U : Opens α) : IsOpen (U : Set α) :=
  U.is_open'


@[simp] theorem mk_coe (U : Opens α) : mk (↑U) U.isOpen = U := rfl


/-- See Note [custom simps projection]. -/
def Simps.coe (U : Opens α) : Set α := U


/-- The interior of a set, as an element of `Opens`. -/
@[simps]
protected def interior (s : Set α) : Opens α :=
  ⟨interior s, isOpen_interior⟩


@[simp]
theorem mem_interior {s : Set α} {x : α} : x ∈ Opens.interior s ↔ x ∈ _root_.interior s := .rfl


theorem gc : GaloisConnection ((↑) : Opens α → Set α) Opens.interior := fun U _ =>
  ⟨fun h => interior_maximal h U.isOpen, fun h => le_trans h interior_subset⟩


/-- The galois coinsertion between sets and opens. -/
def gi : GaloisCoinsertion (↑) (@Opens.interior α _) where
  choice s hs := ⟨s, interior_eq_iff_isOpen.mp <| le_antisymm interior_subset hs⟩
  gc := gc
  u_l_le _ := interior_subset
  choice_eq _s hs := le_antisymm hs interior_subset


instance : CompleteLattice (Opens α) :=
  CompleteLattice.copy (GaloisCoinsertion.liftCompleteLattice gi)
    -- le
    (fun U V => (U : Set α) ⊆ V) rfl
    -- top
    ⟨univ, isOpen_univ⟩ (ext interior_univ.symm)
    -- bot
    ⟨∅, isOpen_empty⟩ rfl
    -- sup
    (fun U V => ⟨↑U ∪ ↑V, U.2.union V.2⟩) rfl
    -- inf
    (fun U V => ⟨↑U ∩ ↑V, U.2.inter V.2⟩)
    (funext₂ fun U V => ext (U.2.inter V.2).interior_eq.symm)
    -- sSup
    (fun S => ⟨⋃ s ∈ S, ↑s, isOpen_biUnion fun s _ => s.2⟩)
    (funext fun _ => ext sSup_image.symm)
    -- sInf
    _ rfl


@[simp]
theorem mk_inf_mk {U V : Set α} {hU : IsOpen U} {hV : IsOpen V} :
    (⟨U, hU⟩ ⊓ ⟨V, hV⟩ : Opens α) = ⟨U ⊓ V, IsOpen.inter hU hV⟩ :=
  rfl


@[simp, norm_cast]
theorem coe_inf (s t : Opens α) : (↑(s ⊓ t) : Set α) = ↑s ∩ ↑t :=
  rfl


@[simp, norm_cast]
theorem coe_sup (s t : Opens α) : (↑(s ⊔ t) : Set α) = ↑s ∪ ↑t :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : Opens α) : Set α) = ∅ :=
  rfl


@[simp] theorem mk_empty : (⟨∅, isOpen_empty⟩ : Opens α) = ⊥ := rfl


@[simp, norm_cast]
theorem coe_eq_empty {U : Opens α} : (U : Set α) = ∅ ↔ U = ⊥ :=
  SetLike.coe_injective.eq_iff' rfl


@[simp]
lemma mem_top (x : α) : x ∈ (⊤ : Opens α) := trivial


@[simp, norm_cast]
theorem coe_top : ((⊤ : Opens α) : Set α) = Set.univ :=
  rfl


@[simp] theorem mk_univ : (⟨univ, isOpen_univ⟩ : Opens α) = ⊤ := rfl


@[simp, norm_cast]
theorem coe_eq_univ {U : Opens α} : (U : Set α) = univ ↔ U = ⊤ :=
  SetLike.coe_injective.eq_iff' rfl


@[simp, norm_cast]
theorem coe_sSup {S : Set (Opens α)} : (↑(sSup S) : Set α) = ⋃ i ∈ S, ↑i :=
  rfl


@[simp, norm_cast]
theorem coe_finset_sup (f : ι → Opens α) (s : Finset ι) : (↑(s.sup f) : Set α) = s.sup ((↑) ∘ f) :=
  map_finset_sup (⟨⟨(↑), coe_sup⟩, coe_bot⟩ : SupBotHom (Opens α) (Set α)) _ _


@[simp, norm_cast]
theorem coe_finset_inf (f : ι → Opens α) (s : Finset ι) : (↑(s.inf f) : Set α) = s.inf ((↑) ∘ f) :=
  map_finset_inf (⟨⟨(↑), coe_inf⟩, coe_top⟩ : InfTopHom (Opens α) (Set α)) _ _


instance : Inhabited (Opens α) := ⟨⊥⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance [IsEmpty α] : Unique (Opens α) where
  uniq _ := ext <| Subsingleton.elim _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance [Nonempty α] : Nontrivial (Opens α) where
  exists_pair_ne := ⟨⊥, ⊤, mt coe_inj.2 empty_ne_univ⟩


@[simp, norm_cast]
theorem coe_iSup {ι} (s : ι → Opens α) : ((⨆ i, s i : Opens α) : Set α) = ⋃ i, s i := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    ι : Sort u_5
    s : ι → TopologicalSpace.Opens α
    ⊢ Eq (↑(iSup fun i => s i)) (Set.iUnion fun i => ↑(s i))
  -/
  simp [iSup]
  /-
    🎉 no goals
  -/


theorem iSup_def {ι} (s : ι → Opens α) : ⨆ i, s i = ⟨⋃ i, s i, isOpen_iUnion fun i => (s i).2⟩ :=
  ext <| coe_iSup s


@[simp]
theorem iSup_mk {ι} (s : ι → Set α) (h : ∀ i, IsOpen (s i)) :
    (⨆ i, ⟨s i, h i⟩ : Opens α) = ⟨⋃ i, s i, isOpen_iUnion h⟩ :=
  iSup_def _


@[simp]
theorem mem_iSup {ι} {x : α} {s : ι → Opens α} : x ∈ iSup s ↔ ∃ i, x ∈ s i := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    ι : Sort u_5
    x : α
    s : ι → TopologicalSpace.Opens α
    ⊢ Iff (Membership.mem (iSup s) x) (Exists fun i => Membership.mem (s i) x)
  -/
  rw [← SetLike.mem_coe]
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    ι : Sort u_5
    x : α
    s : ι → TopologicalSpace.Opens α
    ⊢ Iff (Membership.mem (↑(iSup s)) x) (Exists fun i => Membership.mem (s i) x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_sSup {Us : Set (Opens α)} {x : α} : x ∈ sSup Us ↔ ∃ u ∈ Us, x ∈ u := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    Us : Set (TopologicalSpace.Opens α)
    x : α
    ⊢ Iff (Membership.mem (SupSet.sSup Us) x) (Exists fun u => And (Membership.mem …
  -/
  simp_rw [sSup_eq_iSup, mem_iSup, exists_prop]
  /-
    🎉 no goals
  -/


/-- Open sets in a topological space form a frame. -/
def frameMinimalAxioms : Frame.MinimalAxioms (Opens α) where
  inf_sSup_le_iSup_inf a s :=
               /-
                 ι : Type u_1
                 α : Type u_2
                 β : Type u_3
                 γ : Type u_4
                 inst✝² : TopologicalSpace α
                 inst✝¹ : TopologicalSpace β
                 inst✝ : TopologicalSpace γ
                 a : TopologicalSpace.Opens α
                 s : Set (TopologicalSpace.Opens α)
                 ⊢ Eq ↑(Min.min a (SupSet.sSup s)) ↑(iSup fun b => iSup fun h => Min.min a b)
               -/
    (ext <| by simp only [coe_inf, coe_iSup, coe_sSup, Set.inter_iUnion₂]).le
               /-
                 🎉 no goals
               -/


instance instFrame : Frame (Opens α) := .ofMinimalAxioms frameMinimalAxioms


theorem isOpenEmbedding' (U : Opens α) : IsOpenEmbedding (Subtype.val : U → α) :=
  U.isOpen.isOpenEmbedding_subtypeVal


@[deprecated (since := "2024-10-18")]
alias openEmbedding' := isOpenEmbedding'


theorem isOpenEmbedding_of_le {U V : Opens α} (i : U ≤ V) :
    IsOpenEmbedding (Set.inclusion <| SetLike.coe_subset_coe.2 i) where
  toIsEmbedding := .inclusion i
  isOpen_range := by
    /-
      α : Type u_2
      inst✝ : TopologicalSpace α
      U V : TopologicalSpace.Opens α
      i : LE.le U V
      ⊢ IsOpen (Set.range (Set.inclusion ⋯))
    -/
    rw [Set.range_inclusion i]
    /-
      α : Type u_2
      inst✝ : TopologicalSpace α
      U V : TopologicalSpace.Opens α
      i : LE.le U V
      ⊢ IsOpen (setOf fun x => Membership.mem ↑U ↑x)
    -/
    exact U.isOpen.preimage continuous_subtype_val
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_of_le := isOpenEmbedding_of_le


theorem not_nonempty_iff_eq_bot (U : Opens α) : ¬Set.Nonempty (U : Set α) ↔ U = ⊥ := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    U : TopologicalSpace.Opens α
    ⊢ Iff (Not (↑U).Nonempty) (Eq U Bot.bot)
  -/
  rw [← coe_inj, coe_bot, ← Set.not_nonempty_iff_eq_empty]
  /-
    🎉 no goals
  -/


theorem ne_bot_iff_nonempty (U : Opens α) : U ≠ ⊥ ↔ Set.Nonempty (U : Set α) := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    U : TopologicalSpace.Opens α
    ⊢ Iff (Ne U Bot.bot) (↑U).Nonempty
  -/
  rw [Ne, ← not_nonempty_iff_eq_bot, not_not]
  /-
    🎉 no goals
  -/


/-- An open set in the indiscrete topology is either empty or the whole space. -/
theorem eq_bot_or_top {α} [t : TopologicalSpace α] (h : t = ⊤) (U : Opens α) : U = ⊥ ∨ U = ⊤ := by
  /-
    α : Type u_5
    t : TopologicalSpace α
    h : Eq t Top.top
    U : TopologicalSpace.Opens α
    ⊢ Or (Eq U Bot.bot) (Eq U Top.top)
  -/
  subst h; letI : TopologicalSpace α := ⊤
  /-
    α : Type u_5
    U : TopologicalSpace.Opens α
    this : TopologicalSpace α := Top.top
    ⊢ Or (Eq U Bot.bot) (Eq U Top.top)
  -/
  rw [← coe_eq_empty, ← coe_eq_univ, ← isOpen_top_iff]
  /-
    α : Type u_5
    U : TopologicalSpace.Opens α
    this : TopologicalSpace α := Top.top
    ⊢ IsOpen ↑U
  -/
  exact U.2
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance [Nonempty α] [Subsingleton α] : IsSimpleOrder (Opens α) where
  eq_bot_or_eq_top := eq_bot_or_top <| Subsingleton.elim _ _


/-- A set of `opens α` is a basis if the set of corresponding sets is a topological basis. -/
def IsBasis (B : Set (Opens α)) : Prop :=
  IsTopologicalBasis (((↑) : _ → Set α) '' B)


theorem isBasis_iff_nbhd {B : Set (Opens α)} :
    IsBasis B ↔ ∀ {U : Opens α} {x}, x ∈ U → ∃ U' ∈ B, x ∈ U' ∧ U' ≤ U := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    B : Set (TopologicalSpace.Opens α)
    ⊢ Iff (TopologicalSpace.Opens.IsBasis B) (∀ {U : TopologicalSpace.Opens α} {x  …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      ⊢ ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun U' …
    -/
  · rintro ⟨sU, hU⟩ x hx
    /-
      case mp.mk
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      ⊢ Exists fun U' => And (Membership.mem B U') (And (Membership.mem U' x) (LE.le …
    -/
    rcases h.mem_nhds_iff.mp (IsOpen.mem_nhds hU hx) with ⟨sV, ⟨⟨V, H₁, H₂⟩, hsV⟩⟩
    /-
      case mp.mk.intro.intro.intro.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      sV : Set α
      hsV : And (Membership.mem sV x) (HasSubset.Subset sV sU)
      V : TopologicalSpace.Opens α
      H₁ : Membership.mem B V
      H₂ : Eq (↑V) sV
      ⊢ Exists fun U' => And (Membership.mem B U') (And (Membership.mem U' x) (LE.le …
    -/
    refine ⟨V, H₁, ?_⟩
    /-
      case mp.mk.intro.intro.intro.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      sV : Set α
      hsV : And (Membership.mem sV x) (HasSubset.Subset sV sU)
      V : TopologicalSpace.Opens α
      H₁ : Membership.mem B V
      H₂ : Eq (↑V) sV
      ⊢ And (Membership.mem V x) (LE.le V { carrier := sU, is_open' := hU })
    -/
    cases V
    /-
      case mp.mk.intro.intro.intro.intro.mk
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      sV : Set α
      hsV : And (Membership.mem sV x) (HasSubset.Subset sV sU)
      carrier✝ : Set α
      is_open'✝ : IsOpen carrier✝
      H₁ : Membership.mem B { carrier := carrier✝, is_open' := is_open'✝ }
      H₂ : Eq (↑{ carrier := carrier✝, is_open' := is_open'✝ }) sV
      ⊢ And (Membership.mem { carrier := carrier✝, is_open' := is_open'✝ } x) (LE.le …
    -/
    dsimp at H₂
    /-
      case mp.mk.intro.intro.intro.intro.mk
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      sV : Set α
      hsV : And (Membership.mem sV x) (HasSubset.Subset sV sU)
      carrier✝ : Set α
      is_open'✝ : IsOpen carrier✝
      H₁ : Membership.mem B { carrier := carrier✝, is_open' := is_open'✝ }
      H₂ : Eq carrier✝ sV
      ⊢ And (Membership.mem { carrier := carrier✝, is_open' := is_open'✝ } x) (LE.le …
    -/
    subst H₂
    /-
      case mp.mk.intro.intro.intro.intro.mk
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : TopologicalSpace.Opens.IsBasis B
      sU : Set α
      hU : IsOpen sU
      x : α
      hx : Membership.mem { carrier := sU, is_open' := hU } x
      carrier✝ : Set α
      is_open'✝ : IsOpen carrier✝
      H₁ : Membership.mem B { carrier := carrier✝, is_open' := is_open'✝ }
      hsV : And (Membership.mem carrier✝ x) (HasSubset.Subset carrier✝ sU)
      ⊢ And (Membership.mem { carrier := carrier✝, is_open' := is_open'✝ } x) (LE.le …
    -/
    exact hsV
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
      ⊢ TopologicalSpace.Opens.IsBasis B
    -/
  · refine isTopologicalBasis_of_isOpen_of_nhds ?_ ?_
      /-
        case mpr.refine_1
        α : Type u_2
        inst✝ : TopologicalSpace α
        B : Set (TopologicalSpace.Opens α)
        h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
        ⊢ ∀ (u : Set α), Membership.mem (Set.image SetLike.coe B) u → IsOpen u
      -/
    · rintro sU ⟨U, -, rfl⟩
      /-
        case mpr.refine_1.intro.intro
        α : Type u_2
        inst✝ : TopologicalSpace α
        B : Set (TopologicalSpace.Opens α)
        h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
        U : TopologicalSpace.Opens α
        ⊢ IsOpen ↑U
      -/
      exact U.2
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        α : Type u_2
        inst✝ : TopologicalSpace α
        B : Set (TopologicalSpace.Opens α)
        h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
        ⊢ ∀ (a : α) (u : Set α), Membership.mem u a → IsOpen u → Exists fun v => And ( …
      -/
    · intro x sU hx hsU
      /-
        case mpr.refine_2
        α : Type u_2
        inst✝ : TopologicalSpace α
        B : Set (TopologicalSpace.Opens α)
        h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
        x : α
        sU : Set α
        hx : Membership.mem sU x
        hsU : IsOpen sU
        ⊢ Exists fun v => And (Membership.mem (Set.image SetLike.coe B) v) (And (Membe …
      -/
      rcases @h ⟨sU, hsU⟩ x hx with ⟨V, hV, H⟩
      /-
        case mpr.refine_2.intro.intro
        α : Type u_2
        inst✝ : TopologicalSpace α
        B : Set (TopologicalSpace.Opens α)
        h : ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun  …
        x : α
        sU : Set α
        hx : Membership.mem sU x
        hsU : IsOpen sU
        V : TopologicalSpace.Opens α
        hV : Membership.mem B V
        H : And (Membership.mem V x) (LE.le V { carrier := sU, is_open' := hsU })
        ⊢ Exists fun v => And (Membership.mem (Set.image SetLike.coe B) v) (And (Membe …
      -/
      exact ⟨V, ⟨V, hV, rfl⟩, H⟩
      /-
        🎉 no goals
      -/


theorem isBasis_iff_cover {B : Set (Opens α)} :
    IsBasis B ↔ ∀ U : Opens α, ∃ Us, Us ⊆ B ∧ U = sSup Us := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    B : Set (TopologicalSpace.Opens α)
    ⊢ Iff (TopologicalSpace.Opens.IsBasis B) (∀ (U : TopologicalSpace.Opens α), Ex …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      ⊢ TopologicalSpace.Opens.IsBasis B → ∀ (U : TopologicalSpace.Opens α), Exists  …
    -/
  · intro hB U
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      hB : TopologicalSpace.Opens.IsBasis B
      U : TopologicalSpace.Opens α
      ⊢ Exists fun Us => And (HasSubset.Subset Us B) (Eq U (SupSet.sSup Us))
    -/
    refine ⟨{ V : Opens α | V ∈ B ∧ V ≤ U }, fun U hU => hU.left, ext ?_⟩
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      hB : TopologicalSpace.Opens.IsBasis B
      U : TopologicalSpace.Opens α
      ⊢ Eq ↑U ↑(SupSet.sSup (setOf fun V => And (Membership.mem B V) (LE.le V U)))
    -/
    rw [coe_sSup, hB.open_eq_sUnion' U.isOpen]
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      hB : TopologicalSpace.Opens.IsBasis B
      U : TopologicalSpace.Opens α
      ⊢ Eq (setOf fun s => And (Membership.mem (Set.image SetLike.coe B) s) (HasSubs …
    -/
    simp_rw [sUnion_eq_biUnion, iUnion, mem_setOf_eq, iSup_and, iSup_image]
    /-
      case mp
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      hB : TopologicalSpace.Opens.IsBasis B
      U : TopologicalSpace.Opens α
      ⊢ Eq (iSup fun b => iSup fun h => iSup fun h₂ => ↑b) (iSup fun i => iSup fun h …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      ⊢ (∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset Us …
    -/
  · intro h
    /-
      case mpr
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset U …
      ⊢ TopologicalSpace.Opens.IsBasis B
    -/
    rw [isBasis_iff_nbhd]
    /-
      case mpr
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset U …
      ⊢ ∀ {U : TopologicalSpace.Opens α} {x : α}, Membership.mem U x → Exists fun U' …
    -/
    intro U x hx
    /-
      case mpr
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset U …
      U : TopologicalSpace.Opens α
      x : α
      hx : Membership.mem U x
      ⊢ Exists fun U' => And (Membership.mem B U') (And (Membership.mem U' x) (LE.le …
    -/
    rcases h U with ⟨Us, hUs, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset U …
      x : α
      Us : Set (TopologicalSpace.Opens α)
      hUs : HasSubset.Subset Us B
      hx : Membership.mem (SupSet.sSup Us) x
      ⊢ Exists fun U' => And (Membership.mem B U') (And (Membership.mem U' x) (LE.le …
    -/
    rcases mem_sSup.1 hx with ⟨U, Us, xU⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      B : Set (TopologicalSpace.Opens α)
      h : ∀ (U : TopologicalSpace.Opens α), Exists fun Us => And (HasSubset.Subset U …
      x : α
      Us✝ : Set (TopologicalSpace.Opens α)
      hUs : HasSubset.Subset Us✝ B
      hx : Membership.mem (SupSet.sSup Us✝) x
      U : TopologicalSpace.Opens α
      Us : Membership.mem Us✝ U
      xU : Membership.mem U x
      ⊢ Exists fun U' => And (Membership.mem B U') (And (Membership.mem U' x) (LE.le …
    -/
    exact ⟨U, hUs Us, xU, le_sSup Us⟩
    /-
      🎉 no goals
    -/


/-- If `α` has a basis consisting of compact opens, then an open set in `α` is compact open iff
  it is a finite union of some elements in the basis -/
theorem IsBasis.isCompact_open_iff_eq_finite_iUnion {ι : Type*} (b : ι → Opens α)
    (hb : IsBasis (Set.range b)) (hb' : ∀ i, IsCompact (b i : Set α)) (U : Set α) :
    IsCompact U ∧ IsOpen U ↔ ∃ s : Set ι, s.Finite ∧ U = ⋃ i ∈ s, b i := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    ι : Type u_5
    b : ι → TopologicalSpace.Opens α
    hb : TopologicalSpace.Opens.IsBasis (Set.range b)
    hb' : ∀ (i : ι), IsCompact ↑(b i)
    U : Set α
    ⊢ Iff (And (IsCompact U) (IsOpen U)) (Exists fun s => And s.Finite (Eq U (Set. …
  -/
  apply isCompact_open_iff_eq_finite_iUnion_of_isTopologicalBasis fun i : ι => (b i).1
    /-
      case hb
      α : Type u_2
      inst✝ : TopologicalSpace α
      ι : Type u_5
      b : ι → TopologicalSpace.Opens α
      hb : TopologicalSpace.Opens.IsBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact ↑(b i)
      U : Set α
      ⊢ TopologicalSpace.IsTopologicalBasis (Set.range fun i => (b i).carrier)
    -/
  · convert (config := {transparency := .default}) hb
    /-
      case h.e'_3
      α : Type u_2
      inst✝ : TopologicalSpace α
      ι : Type u_5
      b : ι → TopologicalSpace.Opens α
      hb : TopologicalSpace.Opens.IsBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact ↑(b i)
      U : Set α
      ⊢ Eq (Set.range fun i => (b i).carrier) (Set.image SetLike.coe (Set.range b))
    -/
    ext
    /-
      case h.e'_3.h
      α : Type u_2
      inst✝ : TopologicalSpace α
      ι : Type u_5
      b : ι → TopologicalSpace.Opens α
      hb : TopologicalSpace.Opens.IsBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact ↑(b i)
      U x✝ : Set α
      ⊢ Iff (Membership.mem (Set.range fun i => (b i).carrier) x✝) (Membership.mem ( …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case hb'
      α : Type u_2
      inst✝ : TopologicalSpace α
      ι : Type u_5
      b : ι → TopologicalSpace.Opens α
      hb : TopologicalSpace.Opens.IsBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact ↑(b i)
      U : Set α
      ⊢ ∀ (i : ι), IsCompact (b i).carrier
    -/
  · exact hb'
    /-
      🎉 no goals
    -/


lemma IsBasis.le_iff {α} {t₁ t₂ : TopologicalSpace α}
    {Us : Set (Opens α)} (hUs : @IsBasis α t₂ Us) :
    t₁ ≤ t₂ ↔ ∀ U ∈ Us, IsOpen[t₁] U := by
  /-
    α : Type u_5
    t₁ t₂ : TopologicalSpace α
    Us : Set (TopologicalSpace.Opens α)
    hUs : TopologicalSpace.Opens.IsBasis Us
    ⊢ Iff (LE.le t₁ t₂) (∀ (U : TopologicalSpace.Opens α), Membership.mem Us U → I …
  -/
  conv_lhs => rw [hUs.eq_generateFrom]
  /-
    α : Type u_5
    t₁ t₂ : TopologicalSpace α
    Us : Set (TopologicalSpace.Opens α)
    hUs : TopologicalSpace.Opens.IsBasis Us
    ⊢ Iff (LE.le t₁ (TopologicalSpace.generateFrom (Set.image SetLike.coe Us))) (∀ …
  -/
  simp [Set.subset_def, le_generateFrom_iff_subset_isOpen]
  /-
    🎉 no goals
  -/


@[simp]
theorem isCompactElement_iff (s : Opens α) :
    CompleteLattice.IsCompactElement s ↔ IsCompact (s : Set α) := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    s : TopologicalSpace.Opens α
    ⊢ Iff (CompleteLattice.IsCompactElement s) (IsCompact ↑s)
  -/
  rw [isCompact_iff_finite_subcover, CompleteLattice.isCompactElement_iff]
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    s : TopologicalSpace.Opens α
    ⊢ Iff (∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_ …
  -/
  refine ⟨?_, fun H ι U hU => ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      ⊢ (∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_1) → …
    -/
  · introv H hU hU'
    /-
      case refine_1
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_1)  …
      ι : Type u_2
      U : ι → Set α
      hU : ∀ (i : ι), IsOpen (U i)
      hU' : HasSubset.Subset (↑s) (Set.iUnion fun i => U i)
      ⊢ Exists fun t => HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h  …
    -/
    obtain ⟨t, ht⟩ := H ι (fun i => ⟨U i, hU i⟩) (by simpa)
    /-
      case refine_1.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_1)  …
      ι : Type u_2
      U : ι → Set α
      hU : ∀ (i : ι), IsOpen (U i)
      hU' : HasSubset.Subset (↑s) (Set.iUnion fun i => U i)
      t : Finset ι
      ht : LE.le s (t.sup fun i => { carrier := U i, is_open' := ⋯ })
      ⊢ Exists fun t => HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h  …
    -/
    refine ⟨t, Set.Subset.trans ht ?_⟩
    /-
      case refine_1.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_1)  …
      ι : Type u_2
      U : ι → Set α
      hU : ∀ (i : ι), IsOpen (U i)
      hU' : HasSubset.Subset (↑s) (Set.iUnion fun i => U i)
      t : Finset ι
      ht : LE.le s (t.sup fun i => { carrier := U i, is_open' := ⋯ })
      ⊢ HasSubset.Subset (↑(t.sup fun i => { carrier := U i, is_open' := ⋯ })) (Set. …
    -/
    rw [coe_finset_sup, Finset.sup_eq_iSup]
    /-
      case refine_1.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ (ι : Type u_2) (s_1 : ι → TopologicalSpace.Opens α), LE.le s (iSup s_1)  …
      ι : Type u_2
      U : ι → Set α
      hU : ∀ (i : ι), IsOpen (U i)
      hU' : HasSubset.Subset (↑s) (Set.iUnion fun i => U i)
      t : Finset ι
      ht : LE.le s (t.sup fun i => { carrier := U i, is_open' := ⋯ })
      ⊢ HasSubset.Subset (iSup fun a => iSup fun h => Function.comp SetLike.coe (fun …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · obtain ⟨t, ht⟩ :=
      H (fun i => U i) (fun i => (U i).isOpen) (by simpa using show (s : Set α) ⊆ ↑(iSup U) from hU)
    /-
      case refine_2.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ {ι : Type u_2} (U : ι → Set α), (∀ (i : ι), IsOpen (U i)) → HasSubset.Su …
      ι : Type u_2
      U : ι → TopologicalSpace.Opens α
      hU : LE.le s (iSup U)
      t : Finset ι
      ht : HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      ⊢ Exists fun t => LE.le s (t.sup U)
    -/
    refine ⟨t, Set.Subset.trans ht ?_⟩
    /-
      case refine_2.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ {ι : Type u_2} (U : ι → Set α), (∀ (i : ι), IsOpen (U i)) → HasSubset.Su …
      ι : Type u_2
      U : ι → TopologicalSpace.Opens α
      hU : LE.le s (iSup U)
      t : Finset ι
      ht : HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => ↑(U i)) ↑(t.sup U)
    -/
    simp only [Set.iUnion_subset_iff]
    /-
      case refine_2.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ {ι : Type u_2} (U : ι → Set α), (∀ (i : ι), IsOpen (U i)) → HasSubset.Su …
      ι : Type u_2
      U : ι → TopologicalSpace.Opens α
      hU : LE.le s (iSup U)
      t : Finset ι
      ht : HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      ⊢ ∀ (i : ι), Membership.mem t i → HasSubset.Subset ↑(U i) ↑(t.sup U)
    -/
    show ∀ i ∈ t, U i ≤ t.sup U
    /-
      case refine_2.intro
      α : Type u_2
      inst✝ : TopologicalSpace α
      s : TopologicalSpace.Opens α
      H : ∀ {ι : Type u_2} (U : ι → Set α), (∀ (i : ι), IsOpen (U i)) → HasSubset.Su …
      ι : Type u_2
      U : ι → TopologicalSpace.Opens α
      hU : LE.le s (iSup U)
      t : Finset ι
      ht : HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      ⊢ ∀ (i : ι), Membership.mem t i → LE.le (U i) (t.sup U)
    -/
    exact fun i => Finset.le_sup
    /-
      🎉 no goals
    -/


/-- The preimage of an open set, as an open set. -/
def comap (f : C(α, β)) : FrameHom (Opens β) (Opens α) where
  toFun s := ⟨f ⁻¹' s, s.2.preimage f.continuous⟩
                           /-
                             ι : Type u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             inst✝² : TopologicalSpace α
                             inst✝¹ : TopologicalSpace β
                             inst✝ : TopologicalSpace γ
                             f : ContinuousMap α β
                             s : Set (TopologicalSpace.Opens β)
                             ⊢ Eq ↑({ toFun := fun s => { carrier := Set.preimage ⇑f ↑s, is_open' := ⋯ }, m …
                           -/
  map_sSup' s := ext <| by simp only [coe_sSup, preimage_iUnion, biUnion_image, coe_mk]
                           /-
                             🎉 no goals
                           -/
  map_inf' _ _ := rfl
  map_top' := rfl


@[simp]
theorem comap_id : comap (ContinuousMap.id α) = FrameHom.id _ :=
  FrameHom.ext fun _ => ext rfl


theorem comap_mono (f : C(α, β)) {s t : Opens β} (h : s ≤ t) : comap f s ≤ comap f t :=
  OrderHomClass.mono (comap f) h


@[simp]
theorem coe_comap (f : C(α, β)) (U : Opens β) : ↑(comap f U) = f ⁻¹' U :=
  rfl


@[simp]
theorem mem_comap {f : C(α, β)} {U : Opens β} {x : α} : x ∈ comap f U ↔ f x ∈ U := .rfl


protected theorem comap_comp (g : C(β, γ)) (f : C(α, β)) :
    comap (g.comp f) = (comap f).comp (comap g) :=
  rfl


protected theorem comap_comap (g : C(β, γ)) (f : C(α, β)) (U : Opens γ) :
    comap f (comap g U) = comap (g.comp f) U :=
  rfl


theorem comap_injective [T0Space β] : Injective (comap : C(α, β) → FrameHom (Opens β) (Opens α)) :=
  fun f g h =>
  ContinuousMap.ext fun a =>
    Inseparable.eq <|
      inseparable_iff_forall_isOpen.2 fun s hs =>
        have : comap f ⟨s, hs⟩ = comap g ⟨s, hs⟩ := DFunLike.congr_fun h ⟨_, hs⟩
        show a ∈ f ⁻¹' s ↔ a ∈ g ⁻¹' s from Set.ext_iff.1 (coe_inj.2 this) a


/-- A homeomorphism induces an order-preserving equivalence on open sets, by taking comaps. -/
@[simps (config := .asFn) apply]
def _root_.Homeomorph.opensCongr (f : α ≃ₜ β) : Opens α ≃o Opens β where
  toFun := Opens.comap (f.symm : C(β, α))
  invFun := Opens.comap (f : C(α, β))
  left_inv _ := ext <| f.toEquiv.preimage_symm_preimage _
  right_inv _ := ext <| f.toEquiv.symm_preimage_preimage _
  map_rel_iff' := by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : Homeomorph α β
      ⊢ ∀ {a b : TopologicalSpace.Opens α}, Iff (LE.le ({ toFun := ⇑(TopologicalSpac …
    -/
    simp only [← SetLike.coe_subset_coe]; exact f.symm.surjective.preimage_subset_preimage_iff
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem _root_.Homeomorph.opensCongr_symm (f : α ≃ₜ β) : f.opensCongr.symm = f.symm.opensCongr :=
  rfl


instance [Finite α] : Finite (Opens α) :=
  Finite.of_injective _ SetLike.coe_injective


/-- The open neighborhoods of a point. See also `Opens` or `nhds`. -/
structure OpenNhdsOf (x : α) extends Opens α where
  /-- The point `x` belongs to every `U : TopologicalSpace.OpenNhdsOf x`. -/
  mem' : x ∈ carrier


theorem toOpens_injective : Injective (toOpens : OpenNhdsOf x → Opens α)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


instance : SetLike (OpenNhdsOf x) α where
  coe U := U.1
  coe_injective' := SetLike.coe_injective.comp toOpens_injective


instance canLiftSet : CanLift (Set α) (OpenNhdsOf x) (↑) fun s => IsOpen s ∧ x ∈ s :=
  ⟨fun s hs => ⟨⟨⟨s, hs.1⟩, hs.2⟩, rfl⟩⟩


protected theorem mem (U : OpenNhdsOf x) : x ∈ U :=
  U.mem'


protected theorem isOpen (U : OpenNhdsOf x) : IsOpen (U : Set α) :=
  U.is_open'


instance : OrderTop (OpenNhdsOf x) where
  top := ⟨⊤, Set.mem_univ _⟩
  le_top _ := subset_univ _


instance : Inhabited (OpenNhdsOf x) := ⟨⊤⟩

instance : Min (OpenNhdsOf x) := ⟨fun U V => ⟨U.1 ⊓ V.1, U.2, V.2⟩⟩

instance : Max (OpenNhdsOf x) := ⟨fun U V => ⟨U.1 ⊔ V.1, Or.inl U.2⟩⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance [Subsingleton α] : Unique (OpenNhdsOf x) where
  uniq U := SetLike.ext' <| Subsingleton.eq_univ_of_nonempty ⟨x, U.mem⟩


instance : DistribLattice (OpenNhdsOf x) :=
  toOpens_injective.distribLattice _ (fun _ _ => rfl) fun _ _ => rfl


theorem basis_nhds : (𝓝 x).HasBasis (fun _ : OpenNhdsOf x => True) (↑) :=
  (nhds_basis_opens x).to_hasBasis (fun U hU => ⟨⟨⟨U, hU.2⟩, hU.1⟩, trivial, Subset.rfl⟩) fun U _ =>
    ⟨U, ⟨⟨U.mem, U.isOpen⟩, Subset.rfl⟩⟩


/-- Preimage of an open neighborhood of `f x` under a continuous map `f` as a `LatticeHom`. -/
def comap (f : C(α, β)) (x : α) : LatticeHom (OpenNhdsOf (f x)) (OpenNhdsOf x) where
  toFun U := ⟨Opens.comap f U.1, U.mem⟩
  map_sup' _ _ := rfl
  map_inf' _ _ := rfl


