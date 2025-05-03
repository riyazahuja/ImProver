/-- An ideal on an order `P` is a subset of `P` that is
  - nonempty
  - upward directed (any pair of elements in the ideal has an upper bound in the ideal)
  - downward closed (any element less than an element of the ideal is in the ideal). -/
structure Ideal (P) [LE P] extends LowerSet P where
  /-- The ideal is nonempty. -/
  nonempty' : carrier.Nonempty
  /-- The ideal is upward directed. -/
  directed' : DirectedOn (· ≤ ·) carrier

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: remove this configuration and use the default configuration.
-- We keep this to be consistent with Lean 3.

/-- A subset of a preorder `P` is an ideal if it is
  - nonempty
  - upward directed (any pair of elements in the ideal has an upper bound in the ideal)
  - downward closed (any element less than an element of the ideal is in the ideal). -/
@[mk_iff]
structure IsIdeal {P} [LE P] (I : Set P) : Prop where
  /-- The ideal is downward closed. -/
  IsLowerSet : IsLowerSet I
  /-- The ideal is nonempty. -/
  Nonempty : I.Nonempty
  /-- The ideal is upward directed. -/
  Directed : DirectedOn (· ≤ ·) I


/-- Create an element of type `Order.Ideal` from a set satisfying the predicate
`Order.IsIdeal`. -/
def IsIdeal.toIdeal [LE P] {I : Set P} (h : IsIdeal I) : Ideal P :=
  ⟨⟨I, h.IsLowerSet⟩, h.Nonempty, h.Directed⟩


theorem toLowerSet_injective : Injective (toLowerSet : Ideal P → LowerSet P) := fun s t _ ↦ by
  /-
    P : Type u_1
    inst✝ : LE P
    s t : Order.Ideal P
    x✝ : Eq s.toLowerSet t.toLowerSet
    ⊢ Eq s t
  -/
  cases s
  /-
    case mk
    P : Type u_1
    inst✝ : LE P
    t : Order.Ideal P
    toLowerSet✝ : LowerSet P
    nonempty'✝ : toLowerSet✝.carrier.Nonempty
    directed'✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) toLowerSet✝.carrier
    x✝ : Eq { toLowerSet := toLowerSet✝, nonempty' := nonempty'✝, directed' := dir …
    ⊢ Eq { toLowerSet := toLowerSet✝, nonempty' := nonempty'✝, directed' := direct …
  -/
  cases t
  /-
    case mk.mk
    P : Type u_1
    inst✝ : LE P
    toLowerSet✝¹ : LowerSet P
    nonempty'✝¹ : toLowerSet✝¹.carrier.Nonempty
    directed'✝¹ : DirectedOn (fun x1 x2 => LE.le x1 x2) toLowerSet✝¹.carrier
    toLowerSet✝ : LowerSet P
    nonempty'✝ : toLowerSet✝.carrier.Nonempty
    directed'✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) toLowerSet✝.carrier
    x✝ : Eq { toLowerSet := toLowerSet✝¹, nonempty' := nonempty'✝¹, directed' := d …
    ⊢ Eq { toLowerSet := toLowerSet✝¹, nonempty' := nonempty'✝¹, directed' := dire …
  -/
  congr
  /-
    🎉 no goals
  -/


instance : SetLike (Ideal P) P where
  coe s := s.carrier
  coe_injective' _ _ h := toLowerSet_injective <| SetLike.coe_injective h


@[ext]
theorem ext {s t : Ideal P} : (s : Set P) = t → s = t :=
  SetLike.ext'


@[simp]
theorem carrier_eq_coe (s : Ideal P) : s.carrier = s :=
  rfl


@[simp]
theorem coe_toLowerSet (s : Ideal P) : (s.toLowerSet : Set P) = s :=
  rfl


protected theorem lower (s : Ideal P) : IsLowerSet (s : Set P) :=
  s.lower'


protected theorem nonempty (s : Ideal P) : (s : Set P).Nonempty :=
  s.nonempty'


protected theorem directed (s : Ideal P) : DirectedOn (· ≤ ·) (s : Set P) :=
  s.directed'


protected theorem isIdeal (s : Ideal P) : IsIdeal (s : Set P) :=
  ⟨s.lower, s.nonempty, s.directed⟩


theorem mem_compl_of_ge {x y : P} : x ≤ y → x ∈ (I : Set P)ᶜ → y ∈ (I : Set P)ᶜ := fun h ↦
  mt <| I.lower h


/-- The partial ordering by subset inclusion, inherited from `Set P`. -/
instance instPartialOrderIdeal : PartialOrder (Ideal P) :=
  PartialOrder.lift SetLike.coe SetLike.coe_injective


theorem coe_subset_coe : (s : Set P) ⊆ t ↔ s ≤ t :=
  Iff.rfl


theorem coe_ssubset_coe : (s : Set P) ⊂ t ↔ s < t :=
  Iff.rfl


@[trans]
theorem mem_of_mem_of_le {x : P} {I J : Ideal P} : x ∈ I → I ≤ J → x ∈ J :=
  @Set.mem_of_mem_of_subset P x I J


/-- A proper ideal is one that is not the whole set.
    Note that the whole set might not be an ideal. -/
@[mk_iff]
class IsProper (I : Ideal P) : Prop where
  /-- This ideal is not the whole set. -/
  ne_univ : (I : Set P) ≠ univ


theorem isProper_of_not_mem {I : Ideal P} {p : P} (nmem : p ∉ I) : IsProper I :=
  ⟨fun hp ↦ by
    /-
      P : Type u_1
      inst✝ : LE P
      I : Order.Ideal P
      p : P
      nmem : Not (Membership.mem I p)
      hp : Eq (↑I) Set.univ
      ⊢ False
    -/
    have := mem_univ p
    /-
      P : Type u_1
      inst✝ : LE P
      I : Order.Ideal P
      p : P
      nmem : Not (Membership.mem I p)
      hp : Eq (↑I) Set.univ
      this : Membership.mem Set.univ p
      ⊢ False
    -/
    rw [← hp] at this
    /-
      P : Type u_1
      inst✝ : LE P
      I : Order.Ideal P
      p : P
      nmem : Not (Membership.mem I p)
      hp : Eq (↑I) Set.univ
      this : Membership.mem (↑I) p
      ⊢ False
    -/
    exact nmem this⟩
    /-
      🎉 no goals
    -/


/-- An ideal is maximal if it is maximal in the collection of proper ideals.

Note that `IsCoatom` is less general because ideals only have a top element when `P` is directed
and nonempty. -/
@[mk_iff]
class IsMaximal (I : Ideal P) extends IsProper I : Prop where
  /-- This ideal is maximal in the collection of proper ideals. -/
  maximal_proper : ∀ ⦃J : Ideal P⦄, I < J → (J : Set P) = univ


theorem inter_nonempty [IsDirected P (· ≥ ·)] (I J : Ideal P) : (I ∩ J : Set P).Nonempty := by
  /-
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
    I J : Order.Ideal P
    ⊢ (Inter.inter ↑I ↑J).Nonempty
  -/
  obtain ⟨a, ha⟩ := I.nonempty
  /-
    case intro
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
    I J : Order.Ideal P
    a : P
    ha : Membership.mem (↑I) a
    ⊢ (Inter.inter ↑I ↑J).Nonempty
  -/
  obtain ⟨b, hb⟩ := J.nonempty
  /-
    case intro.intro
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
    I J : Order.Ideal P
    a : P
    ha : Membership.mem (↑I) a
    b : P
    hb : Membership.mem (↑J) b
    ⊢ (Inter.inter ↑I ↑J).Nonempty
  -/
  obtain ⟨c, hac, hbc⟩ := exists_le_le a b
  /-
    case intro.intro.intro.intro
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
    I J : Order.Ideal P
    a : P
    ha : Membership.mem (↑I) a
    b : P
    hb : Membership.mem (↑J) b
    c : P
    hac : LE.le c a
    hbc : LE.le c b
    ⊢ (Inter.inter ↑I ↑J).Nonempty
  -/
  exact ⟨c, I.lower hac ha, J.lower hbc hb⟩
  /-
    🎉 no goals
  -/


/-- In a directed and nonempty order, the top ideal of a is `univ`. -/
instance : OrderTop (Ideal P) where
  top := ⟨⊤, univ_nonempty, directedOn_univ⟩
  le_top _ _ _ := LowerSet.mem_top


@[simp]
theorem top_toLowerSet : (⊤ : Ideal P).toLowerSet = ⊤ :=
  rfl


@[simp]
theorem coe_top : ((⊤ : Ideal P) : Set P) = univ :=
  rfl


theorem isProper_of_ne_top (ne_top : I ≠ ⊤) : IsProper I :=
  ⟨fun h ↦ ne_top <| ext h⟩


theorem IsProper.ne_top (_ : IsProper I) : I ≠ ⊤ :=
  fun h ↦ IsProper.ne_univ <| congr_arg SetLike.coe h


theorem _root_.IsCoatom.isProper (hI : IsCoatom I) : IsProper I :=
  isProper_of_ne_top hI.1


theorem isProper_iff_ne_top : IsProper I ↔ I ≠ ⊤ :=
  ⟨fun h ↦ h.ne_top, fun h ↦ isProper_of_ne_top h⟩


theorem IsMaximal.isCoatom (_ : IsMaximal I) : IsCoatom I :=
  ⟨IsMaximal.toIsProper.ne_top, fun _ h ↦ ext <| IsMaximal.maximal_proper h⟩


theorem IsMaximal.isCoatom' [IsMaximal I] : IsCoatom I :=
  IsMaximal.isCoatom ‹_›


theorem _root_.IsCoatom.isMaximal (hI : IsCoatom I) : IsMaximal I :=
                                                              /-
                                                                P : Type u_1
                                                                inst✝² : LE P
                                                                inst✝¹ : IsDirected P fun x1 x2 => LE.le x1 x2
                                                                inst✝ : Nonempty P
                                                                I : Order.Ideal P
                                                                hI : IsCoatom I
                                                                x✝ : Order.Ideal P
                                                                hJ : LT.lt I x✝
                                                                ⊢ Eq (↑x✝) Set.univ
                                                              -/
  { IsCoatom.isProper hI with maximal_proper := fun _ hJ ↦ by simp [hI.2 _ hJ] }
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem isMaximal_iff_isCoatom : IsMaximal I ↔ IsCoatom I :=
  ⟨fun h ↦ h.isCoatom, fun h ↦ IsCoatom.isMaximal h⟩


@[simp]
theorem bot_mem (s : Ideal P) : ⊥ ∈ s :=
  s.lower bot_le s.nonempty'.some_mem


theorem top_of_top_mem (h : ⊤ ∈ I) : I = ⊤ := by
  /-
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : OrderTop P
    I : Order.Ideal P
    h : Membership.mem I Top.top
    ⊢ Eq I Top.top
  -/
  ext
  /-
    case a.h
    P : Type u_1
    inst✝¹ : LE P
    inst✝ : OrderTop P
    I : Order.Ideal P
    h : Membership.mem I Top.top
    x✝ : P
    ⊢ Iff (Membership.mem (↑I) x✝) (Membership.mem (↑Top.top) x✝)
  -/
  exact iff_of_true (I.lower le_top h) trivial
  /-
    🎉 no goals
  -/


theorem IsProper.top_not_mem (hI : IsProper I) : ⊤ ∉ I := fun h ↦ hI.ne_top <| top_of_top_mem h


/-- The smallest ideal containing a given element. -/
@[simps]
def principal (p : P) : Ideal P where
  toLowerSet := LowerSet.Iic p
  nonempty' := nonempty_Iic
  directed' _ hx _ hy := ⟨p, le_rfl, hx, hy⟩


instance [Inhabited P] : Inhabited (Ideal P) :=
  ⟨Ideal.principal default⟩


@[simp]
theorem principal_le_iff : principal x ≤ I ↔ x ∈ I :=
  ⟨fun h ↦ h le_rfl, fun hx _ hy ↦ I.lower hy hx⟩


@[simp]
theorem mem_principal : x ∈ principal y ↔ x ≤ y :=
  Iff.rfl


lemma mem_principal_self : x ∈ principal x :=
  mem_principal.2 (le_refl x)


/-- There is a bottom ideal when `P` has a bottom element. -/
instance : OrderBot (Ideal P) where
  bot := principal ⊥
               /-
                 P : Type u_1
                 inst✝¹ : Preorder P
                 inst✝ : OrderBot P
                 ⊢ ∀ (a : Order.Ideal P), LE.le Bot.bot a
               -/
  bot_le := by simp
               /-
                 🎉 no goals
               -/


@[simp]
theorem principal_bot : principal (⊥ : P) = ⊥ :=
  rfl


@[simp]
theorem principal_top : principal (⊤ : P) = ⊤ :=
  toLowerSet_injective <| LowerSet.Iic_top


/-- A specific witness of `I.directed` when `P` has joins. -/
theorem sup_mem (hx : x ∈ s) (hy : y ∈ s) : x ⊔ y ∈ s :=
  let ⟨_, hz, hx, hy⟩ := s.directed x hx y hy
  s.lower (sup_le hx hy) hz


@[simp]
theorem sup_mem_iff : x ⊔ y ∈ I ↔ x ∈ I ∧ y ∈ I :=
  ⟨fun h ↦ ⟨I.lower le_sup_left h, I.lower le_sup_right h⟩, fun h ↦ sup_mem h.1 h.2⟩


/-- The infimum of two ideals of a co-directed order is their intersection. -/
instance : Min (Ideal P) :=
  ⟨fun I J ↦
    { toLowerSet := I.toLowerSet ⊓ J.toLowerSet
      nonempty' := inter_nonempty I J
                                                                                      /-
                                                                                        P : Type u_1
                                                                                        inst✝¹ : SemilatticeSup P
                                                                                        inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
                                                                                        x✝ : P
                                                                                        I✝ J✝ s t I J : Order.Ideal P
                                                                                        x : P
                                                                                        hx : Membership.mem (Min.min I.toLowerSet J.toLowerSet).carrier x
                                                                                        y : P
                                                                                        hy : Membership.mem (Min.min I.toLowerSet J.toLowerSet).carrier y
                                                                                        ⊢ And ((fun x1 x2 => LE.le x1 x2) x (Max.max x y)) ((fun x1 x2 => LE.le x1 x2) …
                                                                                      -/
      directed' := fun x hx y hy ↦ ⟨x ⊔ y, ⟨sup_mem hx.1 hy.1, sup_mem hx.2 hy.2⟩, by simp⟩ }⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- The supremum of two ideals of a co-directed order is the union of the down sets of the pointwise
supremum of `I` and `J`. -/
instance : Max (Ideal P) :=
  ⟨fun I J ↦
    { carrier := { x | ∃ i ∈ I, ∃ j ∈ J, x ≤ i ⊔ j }
      nonempty' := by
        /-
          P : Type u_1
          inst✝¹ : SemilatticeSup P
          inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
          x : P
          I✝ J✝ s t I J : Order.Ideal P
          ⊢ { carrier := setOf fun x => Exists fun i => And (Membership.mem I i) (Exists …
        -/
        cases' inter_nonempty I J with w h
        /-
          case intro
          P : Type u_1
          inst✝¹ : SemilatticeSup P
          inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
          x : P
          I✝ J✝ s t I J : Order.Ideal P
          w : P
          h : Membership.mem (Inter.inter ↑I ↑J) w
          ⊢ { carrier := setOf fun x => Exists fun i => And (Membership.mem I i) (Exists …
        -/
        exact ⟨w, w, h.1, w, h.2, le_sup_left⟩
        /-
          🎉 no goals
        -/
      directed' := fun x ⟨xi, _, xj, _, _⟩ y ⟨yi, _, yj, _, _⟩ ↦
        ⟨x ⊔ y, ⟨xi ⊔ yi, sup_mem ‹_› ‹_›, xj ⊔ yj, sup_mem ‹_› ‹_›,
            sup_le
              (calc
                x ≤ xi ⊔ xj := ‹_›
                _ ≤ xi ⊔ yi ⊔ (xj ⊔ yj) := sup_le_sup le_sup_left le_sup_left)
              (calc
                y ≤ yi ⊔ yj := ‹_›
                _ ≤ xi ⊔ yi ⊔ (xj ⊔ yj) := sup_le_sup le_sup_right le_sup_right)⟩,
          le_sup_left, le_sup_right⟩
      lower' := fun _ _ h ⟨yi, hi, yj, hj, hxy⟩ ↦ ⟨yi, hi, yj, hj, h.trans hxy⟩ }⟩


instance : Lattice (Ideal P) :=
  { Ideal.instPartialOrderIdeal with
    sup := (· ⊔ ·)
    le_sup_left := fun _ J i hi ↦
      let ⟨w, hw⟩ := J.nonempty
      ⟨i, hi, w, hw, le_sup_left⟩
    le_sup_right := fun I _ j hj ↦
      let ⟨w, hw⟩ := I.nonempty
      ⟨w, hw, j, hj, le_sup_right⟩
    sup_le := fun _ _ K hIK hJK _ ⟨_, hi, _, hj, ha⟩ ↦
      K.lower ha <| sup_mem (mem_of_mem_of_le hi hIK) (mem_of_mem_of_le hj hJK)
    inf := (· ⊓ ·)
    inf_le_left := fun _ _ ↦ inter_subset_left
    inf_le_right := fun _ _ ↦ inter_subset_right
    le_inf := fun _ _ _ ↦ subset_inter }


@[simp]
theorem coe_sup : ↑(s ⊔ t) = { x | ∃ a ∈ s, ∃ b ∈ t, x ≤ a ⊔ b } :=
  rfl

-- Porting note: Modified `s ∩ t` to `↑s ∩ ↑t`.

@[simp]
theorem coe_inf : (↑(s ⊓ t) : Set P) = ↑s ∩ ↑t :=
  rfl


@[simp]
theorem mem_inf : x ∈ I ⊓ J ↔ x ∈ I ∧ x ∈ J :=
  Iff.rfl


@[simp]
theorem mem_sup : x ∈ I ⊔ J ↔ ∃ i ∈ I, ∃ j ∈ J, x ≤ i ⊔ j :=
  Iff.rfl


theorem lt_sup_principal_of_not_mem (hx : x ∉ I) : I < I ⊔ principal x :=
                                        /-
                                          P : Type u_1
                                          inst✝¹ : SemilatticeSup P
                                          inst✝ : IsDirected P fun x1 x2 => GE.ge x1 x2
                                          x : P
                                          I : Order.Ideal P
                                          hx : Not (Membership.mem I x)
                                          h : Eq I (Max.max I (Order.Ideal.principal x))
                                          ⊢ Membership.mem I x
                                        -/
  le_sup_left.lt_of_ne fun h ↦ hx <| by simpa only [left_eq_sup, principal_le_iff] using h
                                        /-
                                          🎉 no goals
                                        -/


instance : InfSet (Ideal P) :=
  ⟨fun S ↦
    { toLowerSet := ⨅ s ∈ S, toLowerSet s
      nonempty' :=
        ⟨⊥, by
          /-
            P : Type u_1
            inst✝¹ : SemilatticeSup P
            inst✝ : OrderBot P
            x : P
            S : Set (Order.Ideal P)
            ⊢ Membership.mem (iInf fun s => iInf fun h => s.toLowerSet).carrier Bot.bot
          -/
          rw [LowerSet.carrier_eq_coe, LowerSet.coe_iInf₂, Set.mem_iInter₂]
          /-
            P : Type u_1
            inst✝¹ : SemilatticeSup P
            inst✝ : OrderBot P
            x : P
            S : Set (Order.Ideal P)
            ⊢ ∀ (i : Order.Ideal P), Membership.mem S i → Membership.mem (↑i.toLowerSet) B …
          -/
          exact fun s _ ↦ s.bot_mem⟩
          /-
            🎉 no goals
          -/
      directed' := fun a ha b hb ↦
        ⟨a ⊔ b,
          ⟨by
            /-
              P : Type u_1
              inst✝¹ : SemilatticeSup P
              inst✝ : OrderBot P
              x : P
              S : Set (Order.Ideal P)
              a : P
              ha : Membership.mem (iInf fun s => iInf fun h => s.toLowerSet).carrier a
              b : P
              hb : Membership.mem (iInf fun s => iInf fun h => s.toLowerSet).carrier b
              ⊢ Membership.mem (iInf fun s => iInf fun h => s.toLowerSet).carrier (Max.max a …
            -/
            rw [LowerSet.carrier_eq_coe, LowerSet.coe_iInf₂, Set.mem_iInter₂] at ha hb ⊢
            /-
              P : Type u_1
              inst✝¹ : SemilatticeSup P
              inst✝ : OrderBot P
              x : P
              S : Set (Order.Ideal P)
              a : P
              ha : ∀ (i : Order.Ideal P), Membership.mem S i → Membership.mem (↑i.toLowerSet …
              b : P
              hb : ∀ (i : Order.Ideal P), Membership.mem S i → Membership.mem (↑i.toLowerSet …
              ⊢ ∀ (i : Order.Ideal P), Membership.mem S i → Membership.mem (↑i.toLowerSet) ( …
            -/
            exact fun s hs ↦ sup_mem (ha _ hs) (hb _ hs), le_sup_left, le_sup_right⟩⟩ }⟩
            /-
              🎉 no goals
            -/


@[simp]
theorem coe_sInf : (↑(sInf S) : Set P) = ⋂ s ∈ S, ↑s :=
  LowerSet.coe_iInf₂ _


@[simp]
theorem mem_sInf : x ∈ sInf S ↔ ∀ s ∈ S, x ∈ s := by
  /-
    P : Type u_1
    inst✝¹ : SemilatticeSup P
    inst✝ : OrderBot P
    x : P
    S : Set (Order.Ideal P)
    ⊢ Iff (Membership.mem (InfSet.sInf S) x) (∀ (s : Order.Ideal P), Membership.me …
  -/
  simp_rw [← SetLike.mem_coe, coe_sInf, mem_iInter₂]
  /-
    🎉 no goals
  -/


instance : CompleteLattice (Ideal P) :=
  { (inferInstance : Lattice (Ideal P)),
    completeLatticeOfInf (Ideal P) fun S ↦ by
      /-
        P : Type u_1
        inst✝¹ : SemilatticeSup P
        inst✝ : OrderBot P
        x : P
        S✝ S : Set (Order.Ideal P)
        ⊢ IsGLB S (InfSet.sInf S)
      -/
      refine ⟨fun s hs ↦ ?_, fun s hs ↦ by rwa [← coe_subset_coe, coe_sInf, subset_iInter₂_iff]⟩
      /-
        P : Type u_1
        inst✝¹ : SemilatticeSup P
        inst✝ : OrderBot P
        x : P
        S✝ S : Set (Order.Ideal P)
        s : Order.Ideal P
        hs : Membership.mem S s
        ⊢ LE.le (InfSet.sInf S) s
      -/
      rw [← coe_subset_coe, coe_sInf]
      /-
        P : Type u_1
        inst✝¹ : SemilatticeSup P
        inst✝ : OrderBot P
        x : P
        S✝ S : Set (Order.Ideal P)
        s : Order.Ideal P
        hs : Membership.mem S s
        ⊢ HasSubset.Subset (Set.iInter fun s => Set.iInter fun h => ↑s) ↑s
      -/
      exact biInter_subset_of_mem hs with }
      /-
        🎉 no goals
      -/


theorem eq_sup_of_le_sup {x i j : P} (hi : i ∈ I) (hj : j ∈ J) (hx : x ≤ i ⊔ j) :
    ∃ i' ∈ I, ∃ j' ∈ J, x = i' ⊔ j' := by
  /-
    P : Type u_1
    inst✝ : DistribLattice P
    I J : Order.Ideal P
    x i j : P
    hi : Membership.mem I i
    hj : Membership.mem J j
    hx : LE.le x (Max.max i j)
    ⊢ Exists fun i' => And (Membership.mem I i') (Exists fun j' => And (Membership …
  -/
  refine ⟨x ⊓ i, I.lower inf_le_right hi, x ⊓ j, J.lower inf_le_right hj, ?_⟩
  calc
    x = x ⊓ (i ⊔ j) := left_eq_inf.mpr hx
    _ = x ⊓ i ⊔ x ⊓ j := inf_sup_left _ _ _


theorem coe_sup_eq : ↑(I ⊔ J) = { x | ∃ i ∈ I, ∃ j ∈ J, x = i ⊔ j } :=
  Set.ext fun _ ↦
    ⟨fun ⟨_, _, _, _, _⟩ ↦ eq_sup_of_le_sup ‹_› ‹_› ‹_›, fun ⟨i, _, j, _, _⟩ ↦
      ⟨i, ‹_›, j, ‹_›, le_of_eq ‹_›⟩⟩


theorem IsProper.not_mem_of_compl_mem (hI : IsProper I) (hxc : xᶜ ∈ I) : x ∉ I := by
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    hxc : Membership.mem I (HasCompl.compl x)
    ⊢ Not (Membership.mem I x)
  -/
  intro hx
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    hxc : Membership.mem I (HasCompl.compl x)
    hx : Membership.mem I x
    ⊢ False
  -/
  apply hI.top_not_mem
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    hxc : Membership.mem I (HasCompl.compl x)
    hx : Membership.mem I x
    ⊢ Membership.mem I Top.top
  -/
  have ht : x ⊔ xᶜ ∈ I := sup_mem ‹_› ‹_›
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    hxc : Membership.mem I (HasCompl.compl x)
    hx : Membership.mem I x
    ht : Membership.mem I (Max.max x (HasCompl.compl x))
    ⊢ Membership.mem I Top.top
  -/
  rwa [sup_compl_eq_top] at ht
  /-
    🎉 no goals
  -/


theorem IsProper.not_mem_or_compl_not_mem (hI : IsProper I) : x ∉ I ∨ xᶜ ∉ I := by
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    ⊢ Or (Not (Membership.mem I x)) (Not (Membership.mem I (HasCompl.compl x)))
  -/
  have h : xᶜ ∈ I → x ∉ I := hI.not_mem_of_compl_mem
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsProper
    h : Membership.mem I (HasCompl.compl x) → Not (Membership.mem I x)
    ⊢ Or (Not (Membership.mem I x)) (Not (Membership.mem I (HasCompl.compl x)))
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- For a preorder `P`, `Cofinal P` is the type of subsets of `P`
  containing arbitrarily large elements. They are the dense sets in
  the topology whose open sets are terminal segments. -/
structure Cofinal (P) [Preorder P] where
  /-- The carrier of a `Cofinal` is the underlying set. -/
  carrier : Set P
  /-- The `Cofinal` contains arbitrarily large elements. -/
  isCofinal : IsCofinal carrier


@[deprecated Cofinal.isCofinal (since := "2024-12-02")]
alias Cofinal.mem_gt := Cofinal.isCofinal


instance : Inhabited (Cofinal P) :=
  ⟨_, .univ⟩


instance : Membership P (Cofinal P) :=
  ⟨fun D x ↦ x ∈ D.carrier⟩


/-- A (noncomputable) element of a cofinal set lying above a given element. -/
noncomputable def above : P :=
  Classical.choose <| D.isCofinal x


theorem above_mem : D.above x ∈ D :=
  (Classical.choose_spec <| D.isCofinal x).1


theorem le_above : x ≤ D.above x :=
  (Classical.choose_spec <| D.isCofinal x).2


/-- Given a starting point, and a countable family of cofinal sets,
  this is an increasing sequence that intersects each cofinal set. -/
noncomputable def sequenceOfCofinals : ℕ → P
  | 0 => p
  | n + 1 =>
    match Encodable.decode n with
    | none => sequenceOfCofinals n
    | some i => (𝒟 i).above (sequenceOfCofinals n)


theorem sequenceOfCofinals.monotone : Monotone (sequenceOfCofinals p 𝒟) := by
  /-
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    ⊢ Monotone (Order.sequenceOfCofinals p 𝒟)
  -/
  apply monotone_nat_of_le_succ
  /-
    case hf
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    ⊢ ∀ (n : Nat), LE.le (Order.sequenceOfCofinals p 𝒟 n) (Order.sequenceOfCofinal …
  -/
  intro n
  /-
    case hf
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    n : Nat
    ⊢ LE.le (Order.sequenceOfCofinals p 𝒟 n) (Order.sequenceOfCofinals p 𝒟 (HAdd.h …
  -/
  dsimp only [sequenceOfCofinals, Nat.add]
  /-
    case hf
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    n : Nat
    ⊢ LE.le (Order.sequenceOfCofinals p 𝒟 n) (Order.sequenceOfCofinals.match_1 (fu …
  -/
  cases (Encodable.decode n : Option ι)
    /-
      case hf.none
      P : Type u_1
      inst✝¹ : Preorder P
      p : P
      ι : Type u_2
      inst✝ : Encodable ι
      𝒟 : ι → Order.Cofinal P
      n : Nat
      ⊢ LE.le (Order.sequenceOfCofinals p 𝒟 n) (Order.sequenceOfCofinals.match_1 (fu …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hf.some
      P : Type u_1
      inst✝¹ : Preorder P
      p : P
      ι : Type u_2
      inst✝ : Encodable ι
      𝒟 : ι → Order.Cofinal P
      n : Nat
      val✝ : ι
      ⊢ LE.le (Order.sequenceOfCofinals p 𝒟 n) (Order.sequenceOfCofinals.match_1 (fu …
    -/
  · apply Cofinal.le_above
    /-
      🎉 no goals
    -/


theorem sequenceOfCofinals.encode_mem (i : ι) :
    sequenceOfCofinals p 𝒟 (Encodable.encode i + 1) ∈ 𝒟 i := by
  /-
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    i : ι
    ⊢ Membership.mem (𝒟 i) (Order.sequenceOfCofinals p 𝒟 (HAdd.hAdd (Encodable.enc …
  -/
  dsimp only [sequenceOfCofinals, Nat.add]
  /-
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    i : ι
    ⊢ Membership.mem (𝒟 i) (Order.sequenceOfCofinals.match_1 (fun x => P) (Encodab …
  -/
  rw [Encodable.encodek]
  /-
    P : Type u_1
    inst✝¹ : Preorder P
    p : P
    ι : Type u_2
    inst✝ : Encodable ι
    𝒟 : ι → Order.Cofinal P
    i : ι
    ⊢ Membership.mem (𝒟 i) (Order.sequenceOfCofinals.match_1 (fun x => P) (Option. …
  -/
  apply Cofinal.above_mem
  /-
    🎉 no goals
  -/


/-- Given an element `p : P` and a family `𝒟` of cofinal subsets of a preorder `P`,
  indexed by a countable type, `idealOfCofinals p 𝒟` is an ideal in `P` which
  - contains `p`, according to `mem_idealOfCofinals p 𝒟`, and
  - intersects every set in `𝒟`, according to `cofinal_meets_idealOfCofinals p 𝒟`.

  This proves the Rasiowa–Sikorski lemma. -/
def idealOfCofinals : Ideal P where
  carrier := { x : P | ∃ n, x ≤ sequenceOfCofinals p 𝒟 n }
  lower' := fun _ _ hxy ⟨n, hn⟩ ↦ ⟨n, le_trans hxy hn⟩
  nonempty' := ⟨p, 0, le_rfl⟩
  directed' := fun _ ⟨n, hn⟩ _ ⟨m, hm⟩ ↦
    ⟨_, ⟨max n m, le_rfl⟩, le_trans hn <| sequenceOfCofinals.monotone p 𝒟 (le_max_left _ _),
      le_trans hm <| sequenceOfCofinals.monotone p 𝒟 (le_max_right _ _)⟩


theorem mem_idealOfCofinals : p ∈ idealOfCofinals p 𝒟 :=
  ⟨0, le_rfl⟩


/-- `idealOfCofinals p 𝒟` is `𝒟`-generic. -/
theorem cofinal_meets_idealOfCofinals (i : ι) : ∃ x : P, x ∈ 𝒟 i ∧ x ∈ idealOfCofinals p 𝒟 :=
  ⟨_, sequenceOfCofinals.encode_mem p 𝒟 i, _, le_rfl⟩


/-- A non-empty directed union of ideals of sets in a preorder is an ideal. -/
lemma isIdeal_sUnion_of_directedOn {C : Set (Set P)} (hidl : ∀ I ∈ C, IsIdeal I)
    (hD : DirectedOn (· ⊆ ·) C) (hNe : C.Nonempty) : IsIdeal C.sUnion := by
  refine ⟨isLowerSet_sUnion (fun I hI ↦ (hidl I hI).1), Set.nonempty_sUnion.2 ?_,
    directedOn_sUnion hD (fun J hJ => (hidl J hJ).3)⟩
  /-
    P : Type u_1
    inst✝ : Preorder P
    C : Set (Set P)
    hidl : ∀ (I : Set P), Membership.mem C I → Order.IsIdeal I
    hD : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) C
    hNe : C.Nonempty
    ⊢ Exists fun s => And (Membership.mem C s) s.Nonempty
  -/
  let ⟨I, hI⟩ := hNe
  /-
    P : Type u_1
    inst✝ : Preorder P
    C : Set (Set P)
    hidl : ∀ (I : Set P), Membership.mem C I → Order.IsIdeal I
    hD : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) C
    hNe : C.Nonempty
    I : Set P
    hI : Membership.mem C I
    ⊢ Exists fun s => And (Membership.mem C s) s.Nonempty
  -/
  exact ⟨I, ⟨hI, (hidl I hI).2⟩⟩
  /-
    🎉 no goals
  -/


/-- A union of a nonempty chain of ideals of sets is an ideal. -/
lemma isIdeal_sUnion_of_isChain {C : Set (Set P)} (hidl : ∀ I ∈ C, IsIdeal I)
    (hC : IsChain (· ⊆ ·) C) (hNe : C.Nonempty) : IsIdeal C.sUnion :=
  isIdeal_sUnion_of_directedOn hidl hC.directedOn hNe


