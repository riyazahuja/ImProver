/-- A relation on `α` and `β`, aka a set-valued function, aka a partial multifunction -/
def Rel (α β : Type*) :=
  α → β → Prop -- deriving CompleteLattice, Inhabited

-- Porting note: `deriving` above doesn't work.

instance : CompleteLattice (Rel α β) := show CompleteLattice (α → β → Prop) from inferInstance

instance : Inhabited (Rel α β) := show Inhabited (α → β → Prop) from inferInstance


@[ext] theorem ext {r s : Rel α β} : (∀ a, r a = s a) → r = s := funext


/-- The inverse relation : `r.inv x y ↔ r y x`. Note that this is *not* a groupoid inverse. -/
def inv : Rel β α :=
  flip r


theorem inv_def (x : α) (y : β) : r.inv y x ↔ r x y :=
  Iff.rfl


theorem inv_inv : inv (inv r) = r := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq r.inv.inv r
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    r : Rel α β
    x : α
    y : β
    ⊢ Iff (r.inv.inv x y) (r x y)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Domain of a relation -/
def dom := { x | ∃ y, r x y }


theorem dom_mono {r s : Rel α β} (h : r ≤ s) : dom r ⊆ dom s := fun a ⟨b, hx⟩ => ⟨b, h a b hx⟩


/-- Codomain aka range of a relation -/
def codom := { y | ∃ x, r x y }


theorem codom_inv : r.inv.codom = r.dom := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq r.inv.codom r.dom
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    x : α
    ⊢ Iff (Membership.mem r.inv.codom x) (Membership.mem r.dom x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dom_inv : r.inv.dom = r.codom := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq r.inv.dom r.codom
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    x : β
    ⊢ Iff (Membership.mem r.inv.dom x) (Membership.mem r.codom x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Composition of relation; note that it follows the `CategoryTheory/` order of arguments. -/
def comp (r : Rel α β) (s : Rel β γ) : Rel α γ := fun x z => ∃ y, r x y ∧ s y z

-- Porting note: the original `∘` syntax can't be overloaded here, lean considers it ambiguous.

/-- Local syntax for composition of relations. -/
local infixr:90 " • " => Rel.comp


theorem comp_assoc {δ : Type*} (r : Rel α β) (s : Rel β γ) (t : Rel γ δ) :
    (r • s) • t = r • (s • t) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    r : Rel α β
    s : Rel β γ
    t : Rel γ δ
    ⊢ Eq ((r.comp s).comp t) (r.comp (s.comp t))
  -/
  unfold comp; ext (x w); constructor
    /-
      case a.h.a.mp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : Rel α β
      s : Rel β γ
      t : Rel γ δ
      x : α
      w : δ
      ⊢ (Exists fun y => And (Exists fun y_1 => And (r x y_1) (s y_1 y)) (t y w)) →  …
    -/
  · rintro ⟨z, ⟨y, rxy, syz⟩, tzw⟩; exact ⟨y, rxy, z, syz, tzw⟩
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case a.h.a.mpr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : Rel α β
      s : Rel β γ
      t : Rel γ δ
      x : α
      w : δ
      ⊢ (Exists fun y => And (r x y) (Exists fun y_1 => And (s y y_1) (t y_1 w))) →  …
    -/
  · rintro ⟨y, rxy, z, syz, tzw⟩; exact ⟨z, ⟨y, rxy, syz⟩, tzw⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem comp_right_id (r : Rel α β) : r • @Eq β = r := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (r.comp Eq) r
  -/
  unfold comp
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (fun x z => Exists fun y => And (r x y) (Eq y z)) r
  -/
  ext y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    r : Rel α β
    y : α
    x✝ : β
    ⊢ Iff (Exists fun y_1 => And (r y y_1) (Eq y_1 x✝)) (r y x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_left_id (r : Rel α β) : @Eq α • r = r := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (Rel.comp Eq r) r
  -/
  unfold comp
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (fun x z => Exists fun y => And (Eq x y) (r y z)) r
  -/
  ext x
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    r : Rel α β
    x : α
    x✝ : β
    ⊢ Iff (Exists fun y => And (Eq x y) (r y x✝)) (r x x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_right_bot (r : Rel α β) : r • (⊥ : Rel β γ) = ⊥ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    ⊢ Eq (r.comp Bot.bot) Bot.bot
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    x : α
    y : γ
    ⊢ Iff (r.comp Bot.bot x y) (Bot.bot x y)
  -/
  simp [comp, Bot.bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_left_bot (r : Rel α β) : (⊥ : Rel γ α) • r = ⊥ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    ⊢ Eq (Bot.bot.comp r) Bot.bot
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    x : γ
    y : β
    ⊢ Iff (Bot.bot.comp r x y) (Bot.bot x y)
  -/
  simp [comp, Bot.bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_right_top (r : Rel α β) : r • (⊤ : Rel β γ) = fun x _ ↦ x ∈ r.dom := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    ⊢ Eq (r.comp Top.top) fun x x_1 => Membership.mem r.dom x
  -/
  ext x z
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    x : α
    z : γ
    ⊢ Iff (r.comp Top.top x z) (Membership.mem r.dom x)
  -/
  simp [comp, Top.top, dom]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_left_top (r : Rel α β) : (⊤ : Rel γ α) • r = fun _ y ↦ y ∈ r.codom := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    ⊢ Eq (Top.top.comp r) fun x y => Membership.mem r.codom y
  -/
  ext x z
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    x : γ
    z : β
    ⊢ Iff (Top.top.comp r x z) (Membership.mem r.codom z)
  -/
  simp [comp, Top.top, codom]
  /-
    🎉 no goals
  -/


theorem inv_id : inv (@Eq α) = @Eq α := by
  /-
    α : Type u_1
    ⊢ Eq (Rel.inv Eq) Eq
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    x y : α
    ⊢ Iff (Rel.inv Eq x y) (Eq x y)
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> apply Eq.symm
                  /-
                    🎉 no goals
                  -/


theorem inv_comp (r : Rel α β) (s : Rel β γ) : inv (r • s) = inv s • inv r := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    s : Rel β γ
    ⊢ Eq (r.comp s).inv (s.inv.comp r.inv)
  -/
  ext x z
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    s : Rel β γ
    x : γ
    z : α
    ⊢ Iff ((r.comp s).inv x z) (s.inv.comp r.inv x z)
  -/
  simp [comp, inv, flip, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_bot : (⊥ : Rel α β).inv = (⊥ : Rel β α) := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq Bot.bot.inv Bot.bot
  -/
  #adaptation_note /-- nightly-2024-03-16: simp was `simp [Bot.bot, inv, flip]` -/
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq Bot.bot.inv Bot.bot
  -/
  simp [Bot.bot, inv, Function.flip_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_top : (⊤ : Rel α β).inv = (⊤ : Rel β α) := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq Top.top.inv Top.top
  -/
  #adaptation_note /-- nightly-2024-03-16: simp was `simp [Top.top, inv, flip]` -/
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq Top.top.inv Top.top
  -/
  simp [Top.top, inv, Function.flip_def]
  /-
    🎉 no goals
  -/


/-- Image of a set under a relation -/
def image (s : Set α) : Set β := { y | ∃ x ∈ s, r x y }


theorem mem_image (y : β) (s : Set α) : y ∈ image r s ↔ ∃ x ∈ s, r x y :=
  Iff.rfl


theorem image_subset : ((· ⊆ ·) ⇒ (· ⊆ ·)) r.image r.image := fun _ _ h _ ⟨x, xs, rxy⟩ =>
  ⟨x, h xs, rxy⟩


theorem image_mono : Monotone r.image :=
  r.image_subset


theorem image_inter (s t : Set α) : r.image (s ∩ t) ⊆ r.image s ∩ r.image t :=
  r.image_mono.map_inf_le s t


theorem image_union (s t : Set α) : r.image (s ∪ t) = r.image s ∪ r.image t :=
  le_antisymm
    (fun _y ⟨x, xst, rxy⟩ =>
      xst.elim (fun xs => Or.inl ⟨x, ⟨xs, rxy⟩⟩) fun xt => Or.inr ⟨x, ⟨xt, rxy⟩⟩)
    (r.image_mono.le_map_sup s t)


@[simp]
theorem image_id (s : Set α) : image (@Eq α) s = s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Rel.image Eq s) s
  -/
  ext x
  /-
    case h
    α : Type u_1
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (Rel.image Eq s) x) (Membership.mem s x)
  -/
  simp [mem_image]
  /-
    🎉 no goals
  -/


theorem image_comp (s : Rel β γ) (t : Set α) : image (r • s) t = image s (image r t) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    s : Rel β γ
    t : Set α
    ⊢ Eq ((r.comp s).image t) (s.image (r.image t))
  -/
  ext z; simp only [mem_image]; constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : Rel α β
      s : Rel β γ
      t : Set α
      z : γ
      ⊢ (Exists fun x => And (Membership.mem t x) (r.comp s x z)) → Exists fun x =>  …
    -/
  · rintro ⟨x, xt, y, rxy, syz⟩; exact ⟨y, ⟨x, xt, rxy⟩, syz⟩
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : Rel α β
      s : Rel β γ
      t : Set α
      z : γ
      ⊢ (Exists fun x => And (Exists fun x_1 => And (Membership.mem t x_1) (r x_1 x) …
    -/
  · rintro ⟨y, ⟨x, xt, rxy⟩, syz⟩; exact ⟨x, xt, y, rxy, syz⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem image_univ : r.image Set.univ = r.codom := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (r.image Set.univ) r.codom
  -/
  ext y
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    y : β
    ⊢ Iff (Membership.mem (r.image Set.univ) y) (Membership.mem r.codom y)
  -/
  simp [mem_image, codom]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_empty : r.image ∅ = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Eq (r.image EmptyCollection.emptyCollection) EmptyCollection.emptyCollection
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    x : β
    ⊢ Iff (Membership.mem (r.image EmptyCollection.emptyCollection) x) (Membership …
  -/
  simp [mem_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_bot (s : Set α) : (⊥ : Rel α β).image s = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    ⊢ Eq (Bot.bot.image s) EmptyCollection.emptyCollection
  -/
  rw [Set.eq_empty_iff_forall_not_mem]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    ⊢ ∀ (x : β), Not (Membership.mem (Bot.bot.image s) x)
  -/
  intro x h
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    x : β
    h : Membership.mem (Bot.bot.image s) x
    ⊢ False
  -/
  simp [mem_image, Bot.bot] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem image_top {s : Set α} (h : Set.Nonempty s) :
    (⊤ : Rel α β).image s = Set.univ :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              s : Set α
                                              h : s.Nonempty
                                              x✝ : β
                                              ⊢ And (Membership.mem s h.some) (Top.top h.some x✝)
                                            -/
  Set.eq_univ_of_forall fun _ ↦ ⟨h.some, by simp [h.some_mem, Top.top]⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- Preimage of a set under a relation `r`. Same as the image of `s` under `r.inv` -/
def preimage (s : Set β) : Set α :=
  r.inv.image s


theorem mem_preimage (x : α) (s : Set β) : x ∈ r.preimage s ↔ ∃ y ∈ s, r x y :=
  Iff.rfl


theorem preimage_def (s : Set β) : preimage r s = { x | ∃ y ∈ s, r x y } :=
  Set.ext fun _ => mem_preimage _ _ _


theorem preimage_mono {s t : Set β} (h : s ⊆ t) : r.preimage s ⊆ r.preimage t :=
  image_mono _ h


theorem preimage_inter (s t : Set β) : r.preimage (s ∩ t) ⊆ r.preimage s ∩ r.preimage t :=
  image_inter _ s t


theorem preimage_union (s t : Set β) : r.preimage (s ∪ t) = r.preimage s ∪ r.preimage t :=
  image_union _ s t


theorem preimage_id (s : Set α) : preimage (@Eq α) s = s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Rel.preimage Eq s) s
  -/
  simp only [preimage, inv_id, image_id]
  /-
    🎉 no goals
  -/


theorem preimage_comp (s : Rel β γ) (t : Set γ) :
                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           γ : Type u_3
                                                           r : Rel α β
                                                           s : Rel β γ
                                                           t : Set γ
                                                           ⊢ Eq ((r.comp s).preimage t) (r.preimage (s.preimage t))
                                                         -/
    preimage (r • s) t = preimage r (preimage s t) := by simp only [preimage, inv_comp, image_comp]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            r : Rel α β
                                                            ⊢ Eq (r.preimage Set.univ) r.dom
                                                          -/
theorem preimage_univ : r.preimage Set.univ = r.dom := by rw [preimage, image_univ, codom_inv]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  r : Rel α β
                                                  ⊢ Eq (r.preimage EmptyCollection.emptyCollection) EmptyCollection.emptyCollect …
                                                -/
theorem preimage_empty : r.preimage ∅ = ∅ := by rw [preimage, image_empty]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        r : Rel α β
                                                                        s : Set α
                                                                        ⊢ Eq (r.inv.preimage s) (r.image s)
                                                                      -/
theorem preimage_inv (s : Set α) : r.inv.preimage s = r.image s := by rw [preimage, inv_inv]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem preimage_bot (s : Set β) : (⊥ : Rel α β).preimage s = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set β
    ⊢ Eq (Bot.bot.preimage s) EmptyCollection.emptyCollection
  -/
  rw [preimage, inv_bot, image_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_top {s : Set β} (h : Set.Nonempty s) :
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                s : Set β
                                                h : s.Nonempty
                                                ⊢ Eq (Top.top.preimage s) Set.univ
                                              -/
    (⊤ : Rel α β).preimage s = Set.univ := by rwa [← inv_top, preimage, inv_inv, image_top]
                                              /-
                                                🎉 no goals
                                              -/


theorem image_eq_dom_of_codomain_subset {s : Set β} (h : r.codom ⊆ s) : r.preimage s = r.dom := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set β
    h : HasSubset.Subset r.codom s
    ⊢ Eq (r.preimage s) r.dom
  -/
  rw [← preimage_univ]
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set β
    h : HasSubset.Subset r.codom s
    ⊢ Eq (r.preimage s) (r.preimage Set.univ)
  -/
  apply Set.eq_of_subset_of_subset
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      ⊢ HasSubset.Subset (r.preimage s) (r.preimage Set.univ)
    -/
  · exact image_subset _ (Set.subset_univ _)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      ⊢ HasSubset.Subset (r.preimage Set.univ) (r.preimage s)
    -/
  · intro x hx
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      x : α
      hx : Membership.mem (r.preimage Set.univ) x
      ⊢ Membership.mem (r.preimage s) x
    -/
    simp only [mem_preimage, Set.mem_univ, true_and] at hx
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      x : α
      hx : Exists fun y => r x y
      ⊢ Membership.mem (r.preimage s) x
    -/
    rcases hx with ⟨y, ryx⟩
    /-
      case a.intro
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      x : α
      y : β
      ryx : r x y
      ⊢ Membership.mem (r.preimage s) x
    -/
    have hy : y ∈ s := h ⟨x, ryx⟩
    /-
      case a.intro
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set β
      h : HasSubset.Subset r.codom s
      x : α
      y : β
      ryx : r x y
      hy : Membership.mem s y
      ⊢ Membership.mem (r.preimage s) x
    -/
    exact ⟨y, ⟨hy, ryx⟩⟩
    /-
      🎉 no goals
    -/


theorem preimage_eq_codom_of_domain_subset {s : Set α} (h : r.dom ⊆ s) : r.image s = r.codom := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    h : HasSubset.Subset r.dom s
    ⊢ Eq (r.image s) r.codom
  -/
  apply r.inv.image_eq_dom_of_codomain_subset (by rwa [← codom_inv] at h)
  /-
    🎉 no goals
  -/


theorem image_inter_dom_eq (s : Set α) : r.image (s ∩ r.dom) = r.image s := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    ⊢ Eq (r.image (Inter.inter s r.dom)) (r.image s)
  -/
  apply Set.eq_of_subset_of_subset
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      ⊢ HasSubset.Subset (r.image (Inter.inter s r.dom)) (r.image s)
    -/
  · apply r.image_mono (by simp)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      ⊢ HasSubset.Subset (r.image s) (r.image (Inter.inter s r.dom))
    -/
  · intro x h
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      h : Membership.mem (r.image s) x
      ⊢ Membership.mem (r.image (Inter.inter s r.dom)) x
    -/
    rw [mem_image] at *
    /-
      case a
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      h : Exists fun x_1 => And (Membership.mem s x_1) (r x_1 x)
      ⊢ Exists fun x_1 => And (Membership.mem (Inter.inter s r.dom) x_1) (r x_1 x)
    -/
    rcases h with ⟨y, hy, ryx⟩
    /-
      case a.intro.intro
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      y : α
      hy : Membership.mem s y
      ryx : r y x
      ⊢ Exists fun x_1 => And (Membership.mem (Inter.inter s r.dom) x_1) (r x_1 x)
    -/
    use y
    /-
      case h
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      y : α
      hy : Membership.mem s y
      ryx : r y x
      ⊢ And (Membership.mem (Inter.inter s r.dom) y) (r y x)
    -/
    suffices h : y ∈ r.dom by simp_all only [Set.mem_inter_iff, and_self]
    /-
      case h
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      y : α
      hy : Membership.mem s y
      ryx : r y x
      ⊢ Membership.mem r.dom y
    -/
    rw [dom, Set.mem_setOf_eq]
    /-
      case h
      α : Type u_1
      β : Type u_2
      r : Rel α β
      s : Set α
      x : β
      y : α
      hy : Membership.mem s y
      ryx : r y x
      ⊢ Exists fun y_1 => r y y_1
    -/
    use x
    /-
      🎉 no goals
    -/


@[simp]
theorem preimage_inter_codom_eq (s : Set β) : r.preimage (s ∩ r.codom) = r.preimage s := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set β
    ⊢ Eq (r.preimage (Inter.inter s r.codom)) (r.preimage s)
  -/
  rw [← dom_inv, preimage, preimage, image_inter_dom_eq]
  /-
    🎉 no goals
  -/


theorem inter_dom_subset_preimage_image (s : Set α) : s ∩ r.dom ⊆ r.preimage (r.image s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    ⊢ HasSubset.Subset (Inter.inter s r.dom) (r.preimage (r.image s))
  -/
  intro x hx
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    x : α
    hx : Membership.mem (Inter.inter s r.dom) x
    ⊢ Membership.mem (r.preimage (r.image s)) x
  -/
  simp only [Set.mem_inter_iff, dom] at hx
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    x : α
    hx : And (Membership.mem s x) (Membership.mem (setOf fun x => Exists fun y =>  …
    ⊢ Membership.mem (r.preimage (r.image s)) x
  -/
  rcases hx with ⟨hx, ⟨y, rxy⟩⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    x : α
    hx : Membership.mem s x
    y : β
    rxy : r x y
    ⊢ Membership.mem (r.preimage (r.image s)) x
  -/
  use y
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    x : α
    hx : Membership.mem s x
    y : β
    rxy : r x y
    ⊢ And (Membership.mem (r.image s) y) (r.inv y x)
  -/
  simp only [image, Set.mem_setOf_eq]
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set α
    x : α
    hx : Membership.mem s x
    y : β
    rxy : r x y
    ⊢ And (Exists fun x => And (Membership.mem s x) (r x y)) (r.inv y x)
  -/
  exact ⟨⟨x, hx, rxy⟩, rxy⟩
  /-
    🎉 no goals
  -/


theorem image_preimage_subset_inter_codom (s : Set β) : s ∩ r.codom ⊆ r.image (r.preimage s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set β
    ⊢ HasSubset.Subset (Inter.inter s r.codom) (r.image (r.preimage s))
  -/
  rw [← dom_inv, ← preimage_inv]
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    s : Set β
    ⊢ HasSubset.Subset (Inter.inter s r.inv.dom) (r.inv.preimage (r.preimage s))
  -/
  apply inter_dom_subset_preimage_image
  /-
    🎉 no goals
  -/


/-- Core of a set `s : Set β` w.r.t `r : Rel α β` is the set of `x : α` that are related *only*
to elements of `s`. Other generalization of `Function.preimage`. -/
def core (s : Set β) := { x | ∀ y, r x y → y ∈ s }


theorem mem_core (x : α) (s : Set β) : x ∈ r.core s ↔ ∀ y, r x y → y ∈ s :=
  Iff.rfl


theorem core_subset : ((· ⊆ ·) ⇒ (· ⊆ ·)) r.core r.core := fun _s _t h _x h' y rxy => h (h' y rxy)


theorem core_mono : Monotone r.core :=
  r.core_subset


theorem core_inter (s t : Set β) : r.core (s ∩ t) = r.core s ∩ r.core t :=
              /-
                α : Type u_1
                β : Type u_2
                r : Rel α β
                s t : Set β
                ⊢ ∀ (x : α), Iff (Membership.mem (r.core (Inter.inter s t)) x) (Membership.mem …
              -/
  Set.ext (by simp [mem_core, imp_and, forall_and])
              /-
                🎉 no goals
              -/


theorem core_union (s t : Set β) : r.core s ∪ r.core t ⊆ r.core (s ∪ t) :=
  r.core_mono.le_map_sup s t


@[simp]
theorem core_univ : r.core Set.univ = Set.univ :=
              /-
                α : Type u_1
                β : Type u_2
                r : Rel α β
                ⊢ ∀ (x : α), Iff (Membership.mem (r.core Set.univ) x) (Membership.mem Set.univ …
              -/
  Set.ext (by simp [mem_core])
              /-
                🎉 no goals
              -/


                                                       /-
                                                         α : Type u_1
                                                         s : Set α
                                                         ⊢ Eq (Rel.core Eq s) s
                                                       -/
theorem core_id (s : Set α) : core (@Eq α) s = s := by simp [core]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem core_comp (s : Rel β γ) (t : Set γ) : core (r • s) t = core r (core s t) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : Rel α β
    s : Rel β γ
    t : Set γ
    ⊢ Eq ((r.comp s).core t) (r.core (s.core t))
  -/
  ext x; simp only [core, comp, forall_exists_index, and_imp, Set.mem_setOf_eq]; constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : Rel α β
      s : Rel β γ
      t : Set γ
      x : α
      ⊢ (∀ (y : γ) (x_1 : β), r x x_1 → s x_1 y → Membership.mem t y) → ∀ (y : β), r …
    -/
  · exact fun h y rxy z => h z y rxy
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : Rel α β
      s : Rel β γ
      t : Set γ
      x : α
      ⊢ (∀ (y : β), r x y → ∀ (y_1 : γ), s y y_1 → Membership.mem t y_1) → ∀ (y : γ) …
    -/
  · exact fun h z y rzy => h y rzy z
    /-
      🎉 no goals
    -/


/-- Restrict the domain of a relation to a subtype. -/
def restrictDomain (s : Set α) : Rel { x // x ∈ s } β := fun x y => r x.val y


theorem image_subset_iff (s : Set α) (t : Set β) : image r s ⊆ t ↔ s ⊆ core r t :=
  Iff.intro (fun h x xs _y rxy => h ⟨x, xs, rxy⟩) fun h y ⟨_x, xs, rxy⟩ => h xs y rxy


theorem image_core_gc : GaloisConnection r.image r.core :=
  image_subset_iff _


/-- The graph of a function as a relation. -/
def graph (f : α → β) : Rel α β := fun x y => f x = y


@[simp] lemma graph_def (f : α → β) (x y) : f.graph x y ↔ (f x = y) := Iff.rfl


theorem graph_injective : Injective (graph : (α → β) → Rel α β) := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Function.Injective Function.graph
  -/
  intro _ g h
  /-
    α : Type u_1
    β : Type u_2
    a₁✝ g : α → β
    h : Eq (Function.graph a₁✝) (Function.graph g)
    ⊢ Eq a₁✝ g
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    a₁✝ g : α → β
    h : Eq (Function.graph a₁✝) (Function.graph g)
    x : α
    ⊢ Eq (a₁✝ x) (g x)
  -/
  have h2 := congr_fun₂ h x (g x)
  /-
    case h
    α : Type u_1
    β : Type u_2
    a₁✝ g : α → β
    h : Eq (Function.graph a₁✝) (Function.graph g)
    x : α
    h2 : Eq (Function.graph a₁✝ x (g x)) (Function.graph g x (g x))
    ⊢ Eq (a₁✝ x) (g x)
  -/
  simp only [graph_def, eq_iff_iff, iff_true] at h2
  /-
    case h
    α : Type u_1
    β : Type u_2
    a₁✝ g : α → β
    h : Eq (Function.graph a₁✝) (Function.graph g)
    x : α
    h2 : Eq (a₁✝ x) (g x)
    ⊢ Eq (a₁✝ x) (g x)
  -/
  exact h2
  /-
    🎉 no goals
  -/


@[simp] lemma graph_inj {f g : α → β} : f.graph = g.graph ↔ f = g := graph_injective.eq_iff


                                          /-
                                            α : Type u_1
                                            ⊢ Eq (Function.graph id) Eq
                                          -/
theorem graph_id : graph id = @Eq α := by simp (config := { unfoldPartialApp := true }) [graph]
                                          /-
                                            🎉 no goals
                                          -/


theorem graph_comp {f : β → γ} {g : α → β} : graph (f ∘ g) = Rel.comp (graph g) (graph f) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : β → γ
    g : α → β
    ⊢ Eq (Function.graph (Function.comp f g)) ((Function.graph g).comp (Function.g …
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : β → γ
    g : α → β
    x : α
    y : γ
    ⊢ Iff (Function.graph (Function.comp f g) x y) ((Function.graph g).comp (Funct …
  -/
  simp [Rel.comp]
  /-
    🎉 no goals
  -/


theorem Equiv.graph_inv (f : α ≃ β) : (f.symm : β → α).graph = Rel.inv (f : α → β).graph := by
  /-
    α : Type u_1
    β : Type u_2
    f : Equiv α β
    ⊢ Eq (Function.graph ⇑f.symm) (Function.graph ⇑f).inv
  -/
  ext x y
  /-
    case a.h.a
    α : Type u_1
    β : Type u_2
    f : Equiv α β
    x : β
    y : α
    ⊢ Iff (Function.graph (⇑f.symm) x y) ((Function.graph ⇑f).inv x y)
  -/
  aesop (add norm Rel.inv_def)
  /-
    🎉 no goals
  -/


theorem Relation.is_graph_iff (r : Rel α β) : (∃! f, Function.graph f = r) ↔ ∀ x, ∃! y, r x y := by
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Iff (ExistsUnique fun f => Eq (Function.graph f) r) (∀ (x : α), ExistsUnique …
  -/
  unfold Function.graph
  /-
    α : Type u_1
    β : Type u_2
    r : Rel α β
    ⊢ Iff (ExistsUnique fun f => Eq (fun x y => Eq (f x) y) r) (∀ (x : α), ExistsU …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      r : Rel α β
      ⊢ (ExistsUnique fun f => Eq (fun x y => Eq (f x) y) r) → ∀ (x : α), ExistsUniq …
    -/
  · rintro ⟨f, rfl, _⟩ x
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      right✝ : ∀ (y : α → β), (fun f_1 => Eq (fun x y => Eq (f_1 x) y) fun x y => Eq …
      x : α
      ⊢ ExistsUnique fun y => (fun x y => Eq (f x) y) x y
    -/
    use f x
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      right✝ : ∀ (y : α → β), (fun f_1 => Eq (fun x y => Eq (f_1 x) y) fun x y => Eq …
      x : α
      ⊢ And ((fun y => (fun x y => Eq (f x) y) x y) (f x)) (∀ (y : β), (fun y => (fu …
    -/
    simp only [forall_eq', and_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      r : Rel α β
      ⊢ (∀ (x : α), ExistsUnique fun y => r x y) → ExistsUnique fun f => Eq (fun x y …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      r : Rel α β
      h : ∀ (x : α), ExistsUnique fun y => r x y
      ⊢ ExistsUnique fun f => Eq (fun x y => Eq (f x) y) r
    -/
    choose f hf using fun x ↦ (h x).exists
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      r : Rel α β
      h : ∀ (x : α), ExistsUnique fun y => r x y
      f : α → β
      hf : ∀ (x : α), r x (f x)
      ⊢ ExistsUnique fun f => Eq (fun x y => Eq (f x) y) r
    -/
    use f
    /-
      case h
      α : Type u_1
      β : Type u_2
      r : Rel α β
      h : ∀ (x : α), ExistsUnique fun y => r x y
      f : α → β
      hf : ∀ (x : α), r x (f x)
      ⊢ And ((fun f => Eq (fun x y => Eq (f x) y) r) f) (∀ (y : α → β), (fun f => Eq …
    -/
    constructor
      /-
        case h.left
        α : Type u_1
        β : Type u_2
        r : Rel α β
        h : ∀ (x : α), ExistsUnique fun y => r x y
        f : α → β
        hf : ∀ (x : α), r x (f x)
        ⊢ (fun f => Eq (fun x y => Eq (f x) y) r) f
      -/
    · ext x _
      /-
        case h.left.a.h.a
        α : Type u_1
        β : Type u_2
        r : Rel α β
        h : ∀ (x : α), ExistsUnique fun y => r x y
        f : α → β
        hf : ∀ (x : α), r x (f x)
        x : α
        x✝ : β
        ⊢ Iff (Eq (f x) x✝) (r x x✝)
      -/
      constructor
        /-
          case h.left.a.h.a.mp
          α : Type u_1
          β : Type u_2
          r : Rel α β
          h : ∀ (x : α), ExistsUnique fun y => r x y
          f : α → β
          hf : ∀ (x : α), r x (f x)
          x : α
          x✝ : β
          ⊢ Eq (f x) x✝ → r x x✝
        -/
      · rintro rfl
        /-
          case h.left.a.h.a.mp
          α : Type u_1
          β : Type u_2
          r : Rel α β
          h : ∀ (x : α), ExistsUnique fun y => r x y
          f : α → β
          hf : ∀ (x : α), r x (f x)
          x : α
          ⊢ r x (f x)
        -/
        exact hf x
        /-
          🎉 no goals
        -/
        /-
          case h.left.a.h.a.mpr
          α : Type u_1
          β : Type u_2
          r : Rel α β
          h : ∀ (x : α), ExistsUnique fun y => r x y
          f : α → β
          hf : ∀ (x : α), r x (f x)
          x : α
          x✝ : β
          ⊢ r x x✝ → Eq (f x) x✝
        -/
      · exact (h x).unique (hf x)
        /-
          🎉 no goals
        -/
      /-
        case h.right
        α : Type u_1
        β : Type u_2
        r : Rel α β
        h : ∀ (x : α), ExistsUnique fun y => r x y
        f : α → β
        hf : ∀ (x : α), r x (f x)
        ⊢ ∀ (y : α → β), (fun f => Eq (fun x y => Eq (f x) y) r) y → Eq y f
      -/
    · rintro _ rfl
      /-
        case h.right
        α : Type u_1
        β : Type u_2
        f y✝ : α → β
        h : ∀ (x : α), ExistsUnique fun y => (fun x y => Eq (y✝ x) y) x y
        hf : ∀ (x : α), (fun x y => Eq (y✝ x) y) x (f x)
        ⊢ Eq y✝ f
      -/
      exact funext hf
      /-
        🎉 no goals
      -/


theorem image_eq (f : α → β) (s : Set α) : f '' s = (Function.graph f).image s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    ⊢ Eq (Set.image f s) ((Function.graph f).image s)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem preimage_eq (f : α → β) (s : Set β) : f ⁻¹' s = (Function.graph f).preimage s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq (Set.preimage f s) ((Function.graph f).preimage s)
  -/
  simp [Set.preimage, Rel.preimage, Rel.inv, flip, Rel.image]
  /-
    🎉 no goals
  -/


theorem preimage_eq_core (f : α → β) (s : Set β) : f ⁻¹' s = (Function.graph f).core s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq (Set.preimage f s) ((Function.graph f).core s)
  -/
  simp [Set.preimage, Rel.core]
  /-
    🎉 no goals
  -/


