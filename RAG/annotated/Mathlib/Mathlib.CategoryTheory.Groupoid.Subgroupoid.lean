/-- A sugroupoid of `C` consists of a choice of arrows for each pair of vertices, closed
under composition and inverses.
-/
@[ext]
structure Subgroupoid (C : Type u) [Groupoid C] where
  arrows : ∀ c d : C, Set (c ⟶ d)
  protected inv : ∀ {c d} {p : c ⟶ d}, p ∈ arrows c d → Groupoid.inv p ∈ arrows d c
  protected mul : ∀ {c d e} {p}, p ∈ arrows c d → ∀ {q}, q ∈ arrows d e → p ≫ q ∈ arrows c e


theorem inv_mem_iff {c d : C} (f : c ⟶ d) :
    Groupoid.inv f ∈ S.arrows d c ↔ f ∈ S.arrows c d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c d : C
    f : Quiver.Hom c d
    ⊢ Iff (Membership.mem (S.arrows d c) (CategoryTheory.Groupoid.inv f)) (Members …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d : C
      f : Quiver.Hom c d
      ⊢ Membership.mem (S.arrows d c) (CategoryTheory.Groupoid.inv f) → Membership.m …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d : C
      f : Quiver.Hom c d
      h : Membership.mem (S.arrows d c) (CategoryTheory.Groupoid.inv f)
      ⊢ Membership.mem (S.arrows c d) f
    -/
    simpa only [inv_eq_inv, IsIso.inv_inv] using S.inv h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d : C
      f : Quiver.Hom c d
      ⊢ Membership.mem (S.arrows c d) f → Membership.mem (S.arrows d c) (CategoryThe …
    -/
  · apply S.inv
    /-
      🎉 no goals
    -/


theorem mul_mem_cancel_left {c d e : C} {f : c ⟶ d} {g : d ⟶ e} (hf : f ∈ S.arrows c d) :
    f ≫ g ∈ S.arrows c e ↔ g ∈ S.arrows d e := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c d e : C
    f : Quiver.Hom c d
    g : Quiver.Hom d e
    hf : Membership.mem (S.arrows c d) f
    ⊢ Iff (Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g)) …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hf : Membership.mem (S.arrows c d) f
      ⊢ Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g) → Mem …
    -/
  · rintro h
    suffices Groupoid.inv f ≫ f ≫ g ∈ S.arrows d e by
      simpa only [inv_eq_inv, IsIso.inv_hom_id_assoc] using this
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hf : Membership.mem (S.arrows c d) f
      h : Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g)
      ⊢ Membership.mem (S.arrows d e) (CategoryTheory.CategoryStruct.comp (CategoryT …
    -/
    apply S.mul (S.inv hf) h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hf : Membership.mem (S.arrows c d) f
      ⊢ Membership.mem (S.arrows d e) g → Membership.mem (S.arrows c e) (CategoryThe …
    -/
  · apply S.mul hf
    /-
      🎉 no goals
    -/


theorem mul_mem_cancel_right {c d e : C} {f : c ⟶ d} {g : d ⟶ e} (hg : g ∈ S.arrows d e) :
    f ≫ g ∈ S.arrows c e ↔ f ∈ S.arrows c d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c d e : C
    f : Quiver.Hom c d
    g : Quiver.Hom d e
    hg : Membership.mem (S.arrows d e) g
    ⊢ Iff (Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g)) …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hg : Membership.mem (S.arrows d e) g
      ⊢ Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g) → Mem …
    -/
  · rintro h
    suffices (f ≫ g) ≫ Groupoid.inv g ∈ S.arrows c d by
      simpa only [inv_eq_inv, IsIso.hom_inv_id, Category.comp_id, Category.assoc] using this
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hg : Membership.mem (S.arrows d e) g
      h : Membership.mem (S.arrows c e) (CategoryTheory.CategoryStruct.comp f g)
      ⊢ Membership.mem (S.arrows c d) (CategoryTheory.CategoryStruct.comp (CategoryT …
    -/
    apply S.mul h (S.inv hg)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      c d e : C
      f : Quiver.Hom c d
      g : Quiver.Hom d e
      hg : Membership.mem (S.arrows d e) g
      ⊢ Membership.mem (S.arrows c d) f → Membership.mem (S.arrows c e) (CategoryThe …
    -/
  · exact fun hf => S.mul hf hg
    /-
      🎉 no goals
    -/


/-- The vertices of `C` on which `S` has non-trivial isotropy -/
def objs : Set C :=
  {c : C | (S.arrows c c).Nonempty}


theorem mem_objs_of_src {c d : C} {f : c ⟶ d} (h : f ∈ S.arrows c d) : c ∈ S.objs :=
  ⟨f ≫ Groupoid.inv f, S.mul h (S.inv h)⟩


theorem mem_objs_of_tgt {c d : C} {f : c ⟶ d} (h : f ∈ S.arrows c d) : d ∈ S.objs :=
  ⟨Groupoid.inv f ≫ f, S.mul (S.inv h) h⟩


theorem id_mem_of_nonempty_isotropy (c : C) : c ∈ objs S → 𝟙 c ∈ S.arrows c c := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c : C
    ⊢ Membership.mem S.objs c → Membership.mem (S.arrows c c) (CategoryTheory.Cate …
  -/
  rintro ⟨γ, hγ⟩
  /-
    case intro
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c : C
    γ : Quiver.Hom c c
    hγ : Membership.mem (S.arrows c c) γ
    ⊢ Membership.mem (S.arrows c c) (CategoryTheory.CategoryStruct.id c)
  -/
  convert S.mul hγ (S.inv hγ)
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c : C
    γ : Quiver.Hom c c
    hγ : Membership.mem (S.arrows c c) γ
    ⊢ Eq (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryStruct.comp  …
  -/
  simp only [inv_eq_inv, IsIso.hom_inv_id]
  /-
    🎉 no goals
  -/


theorem id_mem_of_src {c d : C} {f : c ⟶ d} (h : f ∈ S.arrows c d) : 𝟙 c ∈ S.arrows c c :=
  id_mem_of_nonempty_isotropy S c (mem_objs_of_src S h)


theorem id_mem_of_tgt {c d : C} {f : c ⟶ d} (h : f ∈ S.arrows c d) : 𝟙 d ∈ S.arrows d d :=
  id_mem_of_nonempty_isotropy S d (mem_objs_of_tgt S h)


/-- A subgroupoid seen as a quiver on vertex set `C` -/
def asWideQuiver : Quiver C :=
  ⟨fun c d => Subtype <| S.arrows c d⟩


/-- The coercion of a subgroupoid as a groupoid -/
@[simps comp_coe, simps (config := .lemmasOnly) inv_coe]
instance coe : Groupoid S.objs where
  Hom a b := S.arrows a.val b.val
  id a := ⟨𝟙 a.val, id_mem_of_nonempty_isotropy S a.val a.prop⟩
  comp p q := ⟨p.val ≫ q.val, S.mul p.prop q.prop⟩
  inv p := ⟨Groupoid.inv p.val, S.inv p.prop⟩


@[simp]
theorem coe_inv_coe' {c d : S.objs} (p : c ⟶ d) :
    (CategoryTheory.inv p).val = CategoryTheory.inv p.val := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c d : ↑S.objs
    p : Quiver.Hom c d
    ⊢ Eq (↑(CategoryTheory.inv p)) (CategoryTheory.inv ↑p)
  -/
  simp only [← inv_eq_inv, coe_inv_coe]
  /-
    🎉 no goals
  -/


/-- The embedding of the coerced subgroupoid to its parent -/
def hom : S.objs ⥤ C where
  obj c := c.val
  map f := f.val
  map_id _ := rfl
  map_comp _ _ := rfl


theorem hom.inj_on_objects : Function.Injective (hom S).obj := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    ⊢ Function.Injective S.hom.obj
  -/
  rintro ⟨c, hc⟩ ⟨d, hd⟩ hcd
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    c : C
    hc : Membership.mem S.objs c
    d : C
    hd : Membership.mem S.objs d
    hcd : Eq (S.hom.obj ⟨c, hc⟩) (S.hom.obj ⟨d, hd⟩)
    ⊢ Eq ⟨c, hc⟩ ⟨d, hd⟩
  -/
  simp only [Subtype.mk_eq_mk]; exact hcd
                                /-
                                  🎉 no goals
                                -/


theorem hom.faithful : ∀ c d, Function.Injective fun f : c ⟶ d => (hom S).map f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    ⊢ ∀ (c d : ↑S.objs), Function.Injective fun f => S.hom.map f
  -/
  rintro ⟨c, hc⟩ ⟨d, hd⟩ ⟨f, hf⟩ ⟨g, hg⟩ hfg; exact Subtype.eq hfg
                                              /-
                                                🎉 no goals
                                              -/


/-- The subgroup of the vertex group at `c` given by the subgroupoid -/
def vertexSubgroup {c : C} (hc : c ∈ S.objs) : Subgroup (c ⟶ c) where
  carrier := S.arrows c c
  mul_mem' hf hg := S.mul hf hg
  one_mem' := id_mem_of_nonempty_isotropy _ _ hc
  inv_mem' hf := S.inv hf


/-- The set of all arrows of a subgroupoid, as a set in `Σ c d : C, c ⟶ d`. -/
@[coe] def toSet (S : Subgroupoid C) : Set (Σ c d : C, c ⟶ d) :=
  {F | F.2.2 ∈ S.arrows F.1 F.2.1}


instance : SetLike (Subgroupoid C) (Σ c d : C, c ⟶ d) where
  coe := toSet
                                                    /-
                                                      C : Type u
                                                      inst✝ : CategoryTheory.Groupoid C
                                                      S✝ x✝¹ x✝ : CategoryTheory.Subgroupoid C
                                                      S : (c d : C) → Set (Quiver.Hom c d)
                                                      inv✝¹ : ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem (S c d) p → Membershi …
                                                      mul✝¹ : ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem (S c d) p → ∀ {q :  …
                                                      T : (c d : C) → Set (Quiver.Hom c d)
                                                      inv✝ : ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem (T c d) p → Membership …
                                                      mul✝ : ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem (T c d) p → ∀ {q : Q …
                                                      h : Eq ↑{ arrows := S, inv := inv✝¹, mul := mul✝¹ } ↑{ arrows := T, inv := inv …
                                                      ⊢ Eq { arrows := S, inv := inv✝¹, mul := mul✝¹ } { arrows := T, inv := inv✝, m …
                                                    -/
  coe_injective' := fun ⟨S, _, _⟩ ⟨T, _, _⟩ h => by ext c d f; apply Set.ext_iff.1 h ⟨c, d, f⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem mem_iff (S : Subgroupoid C) (F : Σ c d, c ⟶ d) : F ∈ S ↔ F.2.2 ∈ S.arrows F.1 F.2.1 :=
  Iff.rfl


theorem le_iff (S T : Subgroupoid C) : S ≤ T ↔ ∀ {c d}, S.arrows c d ⊆ T.arrows c d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S T : CategoryTheory.Subgroupoid C
    ⊢ Iff (LE.le S T) (∀ {c d : C}, HasSubset.Subset (S.arrows c d) (T.arrows c d))
  -/
  rw [SetLike.le_def, Sigma.forall]; exact forall_congr' fun c => Sigma.forall
                                     /-
                                       🎉 no goals
                                     -/


instance : Top (Subgroupoid C) :=
  ⟨{  arrows := fun _ _ => Set.univ
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Groupoid C
                  S : CategoryTheory.Subgroupoid C
                  ⊢ ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem ((fun x x_1 => Set.univ)  …
                -/
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Groupoid C
                  S : CategoryTheory.Subgroupoid C
                  ⊢ ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem ((fun x x_1 => Set.univ) c  …
                -/
      mul := by intros; trivial
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
      inv := by intros; trivial }⟩


theorem mem_top {c d : C} (f : c ⟶ d) : f ∈ (⊤ : Subgroupoid C).arrows c d :=
  trivial


theorem mem_top_objs (c : C) : c ∈ (⊤ : Subgroupoid C).objs := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    c : C
    ⊢ Membership.mem Top.top.objs c
  -/
  dsimp [Top.top, objs]
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    c : C
    ⊢ Set.univ.Nonempty
  -/
  simp only [univ_nonempty]
  /-
    🎉 no goals
  -/


instance : Bot (Subgroupoid C) :=
  ⟨{  arrows := fun _ _ => ∅
      mul := False.elim
      inv := False.elim }⟩


instance : Inhabited (Subgroupoid C) :=
  ⟨⊤⟩


instance : Min (Subgroupoid C) :=
  ⟨fun S T =>
    { arrows := fun c d => S.arrows c d ∩ T.arrows c d
      inv := fun hp ↦ ⟨S.inv hp.1, T.inv hp.2⟩
      mul := fun hp _ hq ↦ ⟨S.mul hp.1 hq.1, T.mul hp.2 hq.2⟩ }⟩


instance : InfSet (Subgroupoid C) :=
  ⟨fun s =>
    { arrows := fun c d => ⋂ S ∈ s, Subgroupoid.arrows S c d
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Groupoid C
                           S : CategoryTheory.Subgroupoid C
                           s : Set (CategoryTheory.Subgroupoid C)
                           c✝ d✝ : C
                           p✝ : Quiver.Hom c✝ d✝
                           hp : Membership.mem ((fun c d => Set.iInter fun S => Set.iInter fun h => S.arr …
                           ⊢ Membership.mem ((fun c d => Set.iInter fun S => Set.iInter fun h => S.arrows …
                         -/
      inv := fun hp ↦ by rw [mem_iInter₂] at hp ⊢; exact fun S hS => S.inv (hp S hS)
                                                   /-
                                                     🎉 no goals
                                                   -/
      mul := fun hp _ hq ↦ by
        /-
          C : Type u
          inst✝ : CategoryTheory.Groupoid C
          S : CategoryTheory.Subgroupoid C
          s : Set (CategoryTheory.Subgroupoid C)
          c✝ d✝ e✝ : C
          p✝ : Quiver.Hom c✝ d✝
          hp : Membership.mem ((fun c d => Set.iInter fun S => Set.iInter fun h => S.arr …
          x✝ : Quiver.Hom d✝ e✝
          hq : Membership.mem ((fun c d => Set.iInter fun S => Set.iInter fun h => S.arr …
          ⊢ Membership.mem ((fun c d => Set.iInter fun S => Set.iInter fun h => S.arrows …
        -/
        rw [mem_iInter₂] at hp hq ⊢
        /-
          C : Type u
          inst✝ : CategoryTheory.Groupoid C
          S : CategoryTheory.Subgroupoid C
          s : Set (CategoryTheory.Subgroupoid C)
          c✝ d✝ e✝ : C
          p✝ : Quiver.Hom c✝ d✝
          hp : ∀ (i : CategoryTheory.Subgroupoid C), Membership.mem s i → Membership.mem …
          x✝ : Quiver.Hom d✝ e✝
          hq : ∀ (i : CategoryTheory.Subgroupoid C), Membership.mem s i → Membership.mem …
          ⊢ ∀ (i : CategoryTheory.Subgroupoid C), Membership.mem s i → Membership.mem (i …
        -/
        exact fun S hS => S.mul (hp S hS) (hq S hS) }⟩
        /-
          🎉 no goals
        -/


theorem mem_sInf_arrows {s : Set (Subgroupoid C)} {c d : C} {p : c ⟶ d} :
    p ∈ (sInf s).arrows c d ↔ ∀ S ∈ s, p ∈ S.arrows c d :=
  mem_iInter₂


theorem mem_sInf {s : Set (Subgroupoid C)} {p : Σ c d : C, c ⟶ d} :
    p ∈ sInf s ↔ ∀ S ∈ s, p ∈ S :=
  mem_sInf_arrows


instance : CompleteLattice (Subgroupoid C) :=
  { completeLatticeOfInf (Subgroupoid C) (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        ⊢ ∀ (s : Set (CategoryTheory.Subgroupoid C)), IsGLB s (InfSet.sInf s)
      -/
      refine fun s => ⟨fun S Ss F => ?_, fun T Tl F fT => ?_⟩ <;> simp only [mem_sInf]
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Groupoid C
        S✝ : CategoryTheory.Subgroupoid C
        s : Set (CategoryTheory.Subgroupoid C)
        S : CategoryTheory.Subgroupoid C
        Ss : Membership.mem s S
        F : Sigma fun c => Sigma fun d => Quiver.Hom c d
        ⊢ (∀ (S : CategoryTheory.Subgroupoid C), Membership.mem s S → Membership.mem S …
      -/
      exacts [fun hp => hp S Ss, fun S Ss => Tl Ss fT]) with
      /-
        🎉 no goals
      -/
    bot := ⊥
    bot_le := fun _ => empty_subset _
    top := ⊤
    le_top := fun _ => subset_univ _
    inf := (· ⊓ ·)
    le_inf := fun _ _ _ RS RT _ pR => ⟨RS pR, RT pR⟩
    inf_le_left := fun _ _ _ => And.left
    inf_le_right := fun _ _ _ => And.right }


theorem le_objs {S T : Subgroupoid C} (h : S ≤ T) : S.objs ⊆ T.objs := fun s ⟨γ, hγ⟩ =>
  ⟨γ, @h ⟨s, s, γ⟩ hγ⟩


/-- The functor associated to the embedding of subgroupoids -/
def inclusion {S T : Subgroupoid C} (h : S ≤ T) : S.objs ⥤ T.objs where
  obj s := ⟨s.val, le_objs h s.prop⟩
  map f := ⟨f.val, @h ⟨_, _, f.val⟩ f.prop⟩
  map_id _ := rfl
  map_comp _ _ := rfl


theorem inclusion_inj_on_objects {S T : Subgroupoid C} (h : S ≤ T) :
    Function.Injective (inclusion h).obj := fun ⟨s, hs⟩ ⟨t, ht⟩ => by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S T : CategoryTheory.Subgroupoid C
    h : LE.le S T
    x✝¹ x✝ : ↑S.objs
    s : C
    hs : Membership.mem S.objs s
    t : C
    ht : Membership.mem S.objs t
    ⊢ Eq ((CategoryTheory.Subgroupoid.inclusion h).obj ⟨s, hs⟩) ((CategoryTheory.S …
  -/
  simpa only [inclusion, Subtype.mk_eq_mk] using id
  /-
    🎉 no goals
  -/


theorem inclusion_faithful {S T : Subgroupoid C} (h : S ≤ T) (s t : S.objs) :
    Function.Injective fun f : s ⟶ t => (inclusion h).map f := fun ⟨f, hf⟩ ⟨g, hg⟩ => by
  -- Porting note: was `...; simpa only [Subtype.mk_eq_mk] using id`
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S T : CategoryTheory.Subgroupoid C
    h : LE.le S T
    s t : ↑S.objs
    x✝¹ x✝ : Quiver.Hom s t
    f : Quiver.Hom ↑s ↑t
    hf : Membership.mem (S.arrows ↑s ↑t) f
    g : Quiver.Hom ↑s ↑t
    hg : Membership.mem (S.arrows ↑s ↑t) g
    ⊢ Eq ((fun f => (CategoryTheory.Subgroupoid.inclusion h).map f) ⟨f, hf⟩) ((fun …
  -/
  dsimp only [inclusion]; rw [Subtype.mk_eq_mk, Subtype.mk_eq_mk]; exact id
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem inclusion_refl {S : Subgroupoid C} : inclusion (le_refl S) = 𝟭 S.objs :=
  Functor.hext (fun _ => rfl) fun _ _ _ => HEq.refl _


theorem inclusion_trans {R S T : Subgroupoid C} (k : R ≤ S) (h : S ≤ T) :
    inclusion (k.trans h) = inclusion k ⋙ inclusion h :=
  rfl


theorem inclusion_comp_embedding {S T : Subgroupoid C} (h : S ≤ T) : inclusion h ⋙ T.hom = S.hom :=
  rfl


/-- The family of arrows of the discrete groupoid -/
inductive Discrete.Arrows : ∀ c d : C, (c ⟶ d) → Prop
  | id (c : C) : Discrete.Arrows c c (𝟙 c)


/-- The only arrows of the discrete groupoid are the identity arrows. -/
def discrete : Subgroupoid C where
  arrows c d := {p | Discrete.Arrows c d p}
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              ⊢ ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun p => …
            -/
  inv := by rintro _ _ _ ⟨⟩; simp only [inv_eq_inv, IsIso.inv_id]; constructor
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              ⊢ ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun p  …
            -/
  mul := by rintro _ _ _ _ ⟨⟩ _ ⟨⟩; rw [Category.comp_id]; constructor
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem mem_discrete_iff {c d : C} (f : c ⟶ d) :
    f ∈ discrete.arrows c d ↔ ∃ h : c = d, f = eqToHom h :=
      /-
        C : Type u
        inst✝ : CategoryTheory.Groupoid C
        c d : C
        f : Quiver.Hom c d
        ⊢ Membership.mem (CategoryTheory.Subgroupoid.discrete.arrows c d) f → Exists f …
      -/
                 /-
                   🎉 no goals
                 -/
  ⟨by rintro ⟨⟩; exact ⟨rfl, rfl⟩, by rintro ⟨rfl, rfl⟩; constructor⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- A subgroupoid is wide if its carrier set is all of `C`-/
structure IsWide : Prop where
  wide : ∀ c, 𝟙 c ∈ S.arrows c c


theorem isWide_iff_objs_eq_univ : S.IsWide ↔ S.objs = Set.univ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    ⊢ Iff S.IsWide (Eq S.objs Set.univ)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      ⊢ S.IsWide → Eq S.objs Set.univ
    -/
  · rintro h
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : S.IsWide
      ⊢ Eq S.objs Set.univ
    -/
                           /-
                             🎉 no goals
                           -/
    ext x; constructor <;> simp only [top_eq_univ, mem_univ, imp_true_iff, forall_true_left]
    /-
      case mp.h.mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : S.IsWide
      x : C
      ⊢ Membership.mem S.objs x
    -/
    apply mem_objs_of_src S (h.wide x)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      ⊢ Eq S.objs Set.univ → S.IsWide
    -/
  · rintro h
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : Eq S.objs Set.univ
      ⊢ S.IsWide
    -/
    refine ⟨fun c => ?_⟩
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : Eq S.objs Set.univ
      c : C
      ⊢ Membership.mem (S.arrows c c) (CategoryTheory.CategoryStruct.id c)
    -/
    obtain ⟨γ, γS⟩ := (le_of_eq h.symm : ⊤ ⊆ S.objs) (Set.mem_univ c)
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : Eq S.objs Set.univ
      c : C
      γ : Quiver.Hom c c
      γS : Membership.mem (S.arrows c c) γ
      ⊢ Membership.mem (S.arrows c c) (CategoryTheory.CategoryStruct.id c)
    -/
    exact id_mem_of_src S γS
    /-
      🎉 no goals
    -/


theorem IsWide.id_mem {S : Subgroupoid C} (Sw : S.IsWide) (c : C) : 𝟙 c ∈ S.arrows c c :=
  Sw.wide c


theorem IsWide.eqToHom_mem {S : Subgroupoid C} (Sw : S.IsWide) {c d : C} (h : c = d) :
                                   /-
                                     C : Type u
                                     inst✝ : CategoryTheory.Groupoid C
                                     S : CategoryTheory.Subgroupoid C
                                     Sw : S.IsWide
                                     c d : C
                                     h : Eq c d
                                     ⊢ Membership.mem (S.arrows c d) (CategoryTheory.eqToHom h)
                                   -/
    eqToHom h ∈ S.arrows c d := by cases h; simp only [eqToHom_refl]; apply Sw.id_mem c
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- A subgroupoid is normal if it is wide and satisfies the expected stability under conjugacy. -/
structure IsNormal extends IsWide S : Prop where
  conj : ∀ {c d} (p : c ⟶ d) {γ : c ⟶ c}, γ ∈ S.arrows c c → Groupoid.inv p ≫ γ ≫ p ∈ S.arrows d d


theorem IsNormal.conj' {S : Subgroupoid C} (Sn : IsNormal S) :
    ∀ {c d} (p : d ⟶ c) {γ : c ⟶ c}, γ ∈ S.arrows c c → p ≫ γ ≫ Groupoid.inv p ∈ S.arrows d d :=
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Groupoid C
                     S : CategoryTheory.Subgroupoid C
                     Sn : S.IsNormal
                     c✝ d✝ : C
                     p : Quiver.Hom d✝ c✝
                     γ : Quiver.Hom c✝ c✝
                     hs : Membership.mem (S.arrows c✝ c✝) γ
                     ⊢ Membership.mem (S.arrows d✝ d✝) (CategoryTheory.CategoryStruct.comp p (Categ …
                   -/
  fun p γ hs => by convert Sn.conj (Groupoid.inv p) hs; simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem IsNormal.conjugation_bij (Sn : IsNormal S) {c d} (p : c ⟶ d) :
    Set.BijOn (fun γ : c ⟶ c => Groupoid.inv p ≫ γ ≫ p) (S.arrows c c) (S.arrows d d) := by
  refine ⟨fun γ γS => Sn.conj p γS, fun γ₁ _ γ₂ _ h => ?_, fun δ δS =>
    ⟨p ≫ δ ≫ Groupoid.inv p, Sn.conj' p δS, ?_⟩⟩
  · simpa only [inv_eq_inv, Category.assoc, IsIso.hom_inv_id, Category.comp_id,
      IsIso.hom_inv_id_assoc] using p ≫= h =≫ inv p
  · simp only [inv_eq_inv, Category.assoc, IsIso.inv_hom_id, Category.comp_id,
      IsIso.inv_hom_id_assoc]


theorem top_isNormal : IsNormal (⊤ : Subgroupoid C) :=
  { wide := fun _ => trivial
    conj := fun _ _ _ => trivial }


theorem sInf_isNormal (s : Set <| Subgroupoid C) (sn : ∀ S ∈ s, IsNormal S) : IsNormal (sInf s) :=
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Groupoid C
                 s : Set (CategoryTheory.Subgroupoid C)
                 sn : ∀ (S : CategoryTheory.Subgroupoid C), Membership.mem s S → S.IsNormal
                 ⊢ ∀ (c : C), Membership.mem ((InfSet.sInf s).arrows c c) (CategoryTheory.Categ …
               -/
  { wide := by simp_rw [sInf, mem_iInter₂]; exact fun c S Ss => (sn S Ss).wide c
                                            /-
                                              🎉 no goals
                                            -/
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Groupoid C
                 s : Set (CategoryTheory.Subgroupoid C)
                 sn : ∀ (S : CategoryTheory.Subgroupoid C), Membership.mem s S → S.IsNormal
                 ⊢ ∀ {c d : C} (p : Quiver.Hom c d) {γ : Quiver.Hom c c}, Membership.mem ((InfS …
               -/
    conj := by simp_rw [sInf, mem_iInter₂]; exact fun p γ hγ S Ss => (sn S Ss).conj p (hγ S Ss) }
                                            /-
                                              🎉 no goals
                                            -/


theorem discrete_isNormal : (@discrete C _).IsNormal :=
                        /-
                          C : Type u
                          inst✝ : CategoryTheory.Groupoid C
                          c : C
                          ⊢ Membership.mem (CategoryTheory.Subgroupoid.discrete.arrows c c) (CategoryThe …
                        -/
  { wide := fun c => by constructor
                        /-
                          🎉 no goals
                        -/
    conj := fun f γ hγ => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Groupoid C
        c✝ d✝ : C
        f : Quiver.Hom c✝ d✝
        γ : Quiver.Hom c✝ c✝
        hγ : Membership.mem (CategoryTheory.Subgroupoid.discrete.arrows c✝ c✝) γ
        ⊢ Membership.mem (CategoryTheory.Subgroupoid.discrete.arrows d✝ d✝) (CategoryT …
      -/
      cases hγ
      /-
        case id
        C : Type u
        inst✝ : CategoryTheory.Groupoid C
        c✝ d✝ : C
        f : Quiver.Hom c✝ d✝
        ⊢ Membership.mem (CategoryTheory.Subgroupoid.discrete.arrows d✝ d✝) (CategoryT …
      -/
      simp only [inv_eq_inv, Category.id_comp, IsIso.inv_hom_id]; constructor }
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem IsNormal.vertexSubgroup (Sn : IsNormal S) (c : C) (cS : c ∈ S.objs) :
    (S.vertexSubgroup cS).Normal where
                        /-
                          C : Type u
                          inst✝ : CategoryTheory.Groupoid C
                          S : CategoryTheory.Subgroupoid C
                          Sn : S.IsNormal
                          c : C
                          cS : Membership.mem S.objs c
                          x : Quiver.Hom c c
                          hx : Membership.mem (S.vertexSubgroup cS) x
                          y : Quiver.Hom c c
                          ⊢ Membership.mem (S.vertexSubgroup cS) (HMul.hMul (HMul.hMul y x) (Inv.inv y))
                        -/
  conj_mem x hx y := by rw [mul_assoc]; exact Sn.conj' y hx
                                        /-
                                          🎉 no goals
                                        -/


/-- The subgropoid generated by the set of arrows `X` -/
def generated : Subgroupoid C :=
  sInf {S : Subgroupoid C | ∀ c d, X c d ⊆ S.arrows c d}


theorem subset_generated (c d : C) : X c d ⊆ (generated X).arrows c d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    c d : C
    ⊢ HasSubset.Subset (X c d) ((CategoryTheory.Subgroupoid.generated X).arrows c d)
  -/
  dsimp only [generated, sInf]
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    c d : C
    ⊢ HasSubset.Subset (X c d) (Set.iInter fun S => Set.iInter fun h => S.arrows c …
  -/
  simp only [subset_iInter₂_iff]
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    c d : C
    ⊢ ∀ (i : CategoryTheory.Subgroupoid C), Membership.mem (setOf fun S => ∀ (c d  …
  -/
  exact fun S hS f fS => hS _ _ fS
  /-
    🎉 no goals
  -/


/-- The normal sugroupoid generated by the set of arrows `X` -/
def generatedNormal : Subgroupoid C :=
  sInf {S : Subgroupoid C | (∀ c d, X c d ⊆ S.arrows c d) ∧ S.IsNormal}


theorem generated_le_generatedNormal : generated X ≤ generatedNormal X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    ⊢ LE.le (CategoryTheory.Subgroupoid.generated X) (CategoryTheory.Subgroupoid.g …
  -/
  apply @sInf_le_sInf (Subgroupoid C) _
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    ⊢ HasSubset.Subset (setOf fun S => And (∀ (c d : C), HasSubset.Subset (X c d)  …
  -/
  exact fun S ⟨h, _⟩ => h
  /-
    🎉 no goals
  -/


theorem generatedNormal_isNormal : (generatedNormal X).IsNormal :=
  sInf_isNormal _ fun _ h => h.right


theorem IsNormal.generatedNormal_le {S : Subgroupoid C} (Sn : S.IsNormal) :
    generatedNormal X ≤ S ↔ ∀ c d, X c d ⊆ S.arrows c d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    X : (c d : C) → Set (Quiver.Hom c d)
    S : CategoryTheory.Subgroupoid C
    Sn : S.IsNormal
    ⊢ Iff (LE.le (CategoryTheory.Subgroupoid.generatedNormal X) S) (∀ (c d : C), H …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      ⊢ LE.le (CategoryTheory.Subgroupoid.generatedNormal X) S → ∀ (c d : C), HasSub …
    -/
  · rintro h c d
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      h : LE.le (CategoryTheory.Subgroupoid.generatedNormal X) S
      c d : C
      ⊢ HasSubset.Subset (X c d) (S.arrows c d)
    -/
    have h' := generated_le_generatedNormal X
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      h : LE.le (CategoryTheory.Subgroupoid.generatedNormal X) S
      c d : C
      h' : LE.le (CategoryTheory.Subgroupoid.generated X) (CategoryTheory.Subgroupoi …
      ⊢ HasSubset.Subset (X c d) (S.arrows c d)
    -/
    rw [le_iff] at h h'
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      h : ∀ {c d : C}, HasSubset.Subset ((CategoryTheory.Subgroupoid.generatedNormal …
      c d : C
      h' : ∀ {c d : C}, HasSubset.Subset ((CategoryTheory.Subgroupoid.generated X).a …
      ⊢ HasSubset.Subset (X c d) (S.arrows c d)
    -/
    exact ((subset_generated X c d).trans (@h' c d)).trans (@h c d)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      ⊢ (∀ (c d : C), HasSubset.Subset (X c d) (S.arrows c d)) → LE.le (CategoryTheo …
    -/
  · rintro h
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      h : ∀ (c d : C), HasSubset.Subset (X c d) (S.arrows c d)
      ⊢ LE.le (CategoryTheory.Subgroupoid.generatedNormal X) S
    -/
    apply @sInf_le (Subgroupoid C) _
    /-
      case mpr.a
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X : (c d : C) → Set (Quiver.Hom c d)
      S : CategoryTheory.Subgroupoid C
      Sn : S.IsNormal
      h : ∀ (c d : C), HasSubset.Subset (X c d) (S.arrows c d)
      ⊢ Membership.mem (setOf fun S => And (∀ (c d : C), HasSubset.Subset (X c d) (S …
    -/
    exact ⟨h, Sn⟩
    /-
      🎉 no goals
    -/


/-- A functor between groupoid defines a map of subgroupoids in the reverse direction
by taking preimages.
 -/
def comap (S : Subgroupoid D) : Subgroupoid C where
  arrows c d := {f : c ⟶ d | φ.map f ∈ S.arrows (φ.obj c) (φ.obj d)}
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Groupoid C
                 S✝ : CategoryTheory.Subgroupoid C
                 D : Type u_1
                 inst✝ : CategoryTheory.Groupoid D
                 φ : CategoryTheory.Functor C D
                 S : CategoryTheory.Subgroupoid D
                 c✝ d✝ : C
                 p✝ : Quiver.Hom c✝ d✝
                 hp : Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.ob …
                 ⊢ Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.obj c …
               -/
  inv hp := by rw [mem_setOf, inv_eq_inv, φ.map_inv, ← inv_eq_inv]; exact S.inv hp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  mul := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      S : CategoryTheory.Subgroupoid D
      ⊢ ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun f  …
    -/
    intros
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      S : CategoryTheory.Subgroupoid D
      c✝ d✝ e✝ : C
      p✝ : Quiver.Hom c✝ d✝
      a✝¹ : Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.o …
      q✝ : Quiver.Hom d✝ e✝
      a✝ : Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.ob …
      ⊢ Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.obj c …
    -/
    simp only [mem_setOf, Functor.map_comp]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      S : CategoryTheory.Subgroupoid D
      c✝ d✝ e✝ : C
      p✝ : Quiver.Hom c✝ d✝
      a✝¹ : Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.o …
      q✝ : Quiver.Hom d✝ e✝
      a✝ : Membership.mem ((fun c d => setOf fun f => Membership.mem (S.arrows (φ.ob …
      ⊢ Membership.mem (S.arrows (φ.obj c✝) (φ.obj e✝)) (CategoryTheory.CategoryStru …
    -/
                    /-
                      🎉 no goals
                    -/
    apply S.mul <;> assumption
                    /-
                      🎉 no goals
                    -/


theorem comap_mono (S T : Subgroupoid D) : S ≤ T → comap φ S ≤ comap φ T := fun ST _ =>
  @ST ⟨_, _, _⟩


theorem isNormal_comap {S : Subgroupoid D} (Sn : IsNormal S) : IsNormal (comap φ S) where
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Groupoid C
                 D : Type u_1
                 inst✝ : CategoryTheory.Groupoid D
                 φ : CategoryTheory.Functor C D
                 S : CategoryTheory.Subgroupoid D
                 Sn : S.IsNormal
                 c : C
                 ⊢ Membership.mem ((CategoryTheory.Subgroupoid.comap φ S).arrows c c) (Category …
               -/
  wide c := by rw [comap, mem_setOf, Functor.map_id]; apply Sn.wide
                                                      /-
                                                        🎉 no goals
                                                      -/
  conj f γ hγ := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      S : CategoryTheory.Subgroupoid D
      Sn : S.IsNormal
      c✝ d✝ : C
      f : Quiver.Hom c✝ d✝
      γ : Quiver.Hom c✝ c✝
      hγ : Membership.mem ((CategoryTheory.Subgroupoid.comap φ S).arrows c✝ c✝) γ
      ⊢ Membership.mem ((CategoryTheory.Subgroupoid.comap φ S).arrows d✝ d✝) (Catego …
    -/
    simp_rw [inv_eq_inv f, comap, mem_setOf, Functor.map_comp, Functor.map_inv, ← inv_eq_inv]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      S : CategoryTheory.Subgroupoid D
      Sn : S.IsNormal
      c✝ d✝ : C
      f : Quiver.Hom c✝ d✝
      γ : Quiver.Hom c✝ c✝
      hγ : Membership.mem ((CategoryTheory.Subgroupoid.comap φ S).arrows c✝ c✝) γ
      ⊢ Membership.mem (S.arrows (φ.obj d✝) (φ.obj d✝)) (CategoryTheory.CategoryStru …
    -/
    exact Sn.conj _ hγ
    /-
      🎉 no goals
    -/


@[simp]
theorem comap_comp {E : Type*} [Groupoid E] (ψ : D ⥤ E) : comap (φ ⋙ ψ) = comap φ ∘ comap ψ :=
  rfl


/-- The kernel of a functor between subgroupoid is the preimage. -/
def ker : Subgroupoid C :=
  comap φ discrete


theorem mem_ker_iff {c d : C} (f : c ⟶ d) :
    f ∈ (ker φ).arrows c d ↔ ∃ h : φ.obj c = φ.obj d, φ.map f = eqToHom h :=
  mem_discrete_iff (φ.map f)


theorem ker_isNormal : (ker φ).IsNormal :=
  isNormal_comap φ discrete_isNormal


@[simp]
theorem ker_comp {E : Type*} [Groupoid E] (ψ : D ⥤ E) : ker (φ ⋙ ψ) = comap φ (ker ψ) :=
  rfl


/-- The family of arrows of the image of a subgroupoid under a functor injective on objects -/
inductive Map.Arrows (hφ : Function.Injective φ.obj) (S : Subgroupoid C) : ∀ c d : D, (c ⟶ d) → Prop
  | im {c d : C} (f : c ⟶ d) (hf : f ∈ S.arrows c d) : Map.Arrows hφ S (φ.obj c) (φ.obj d) (φ.map f)


theorem Map.arrows_iff (hφ : Function.Injective φ.obj) (S : Subgroupoid C) {c d : D} (f : c ⟶ d) :
    Map.Arrows φ hφ S c d f ↔
      ∃ (a b : C) (g : a ⟶ b) (ha : φ.obj a = c) (hb : φ.obj b = d) (_hg : g ∈ S.arrows a b),
        f = eqToHom ha.symm ≫ φ.map g ≫ eqToHom hb := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    S : CategoryTheory.Subgroupoid C
    c d : D
    f : Quiver.Hom c d
    ⊢ Iff (CategoryTheory.Subgroupoid.Map.Arrows φ hφ S c d f) (Exists fun a => Ex …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      f : Quiver.Hom c d
      ⊢ CategoryTheory.Subgroupoid.Map.Arrows φ hφ S c d f → Exists fun a => Exists  …
    -/
  · rintro ⟨g, hg⟩; exact ⟨_, _, g, rfl, rfl, hg, eq_conj_eqToHom _⟩
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      f : Quiver.Hom c d
      ⊢ (Exists fun a => Exists fun b => Exists fun g => Exists fun ha => Exists fun …
    -/
  · rintro ⟨a, b, g, rfl, rfl, hg, rfl⟩; rw [← eq_conj_eqToHom]; constructor; exact hg
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- The "forward" image of a subgroupoid under a functor injective on objects -/
def map (hφ : Function.Injective φ.obj) (S : Subgroupoid C) : Subgroupoid D where
  arrows c d := {x | Map.Arrows φ hφ S c d x}
  inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      ⊢ ∀ {c d : D} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun x => …
    -/
    rintro _ _ _ ⟨⟩
    /-
      case im
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      c✝ d✝ : C
      f✝ : Quiver.Hom c✝ d✝
      hf✝ : Membership.mem (S.arrows c✝ d✝) f✝
      ⊢ Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map.Ar …
    -/
    rw [inv_eq_inv, ← Functor.map_inv, ← inv_eq_inv]
    /-
      case im
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      c✝ d✝ : C
      f✝ : Quiver.Hom c✝ d✝
      hf✝ : Membership.mem (S.arrows c✝ d✝) f✝
      ⊢ Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map.Ar …
    -/
    constructor; apply S.inv; assumption
                              /-
                                🎉 no goals
                              -/
  mul := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      ⊢ ∀ {c d e : D} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun x  …
    -/
    rintro _ _ _ _ ⟨f, hf⟩ q hq
    /-
      case im
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      e✝ c d : D
      c✝ d✝ : C
      f : Quiver.Hom c✝ d✝
      hf : Membership.mem (S.arrows c✝ d✝) f
      q : Quiver.Hom (φ.obj d✝) e✝
      hq : Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map …
      ⊢ Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map.Ar …
    -/
    obtain ⟨c₃, c₄, g, he, rfl, hg, gq⟩ := (Map.arrows_iff φ hφ S q).mp hq
    /-
      case im.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      c✝ d✝ : C
      f : Quiver.Hom c✝ d✝
      hf : Membership.mem (S.arrows c✝ d✝) f
      c₃ c₄ : C
      g : Quiver.Hom c₃ c₄
      he : Eq (φ.obj c₃) (φ.obj d✝)
      q : Quiver.Hom (φ.obj d✝) (φ.obj c₄)
      hq : Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map …
      hg : Membership.mem (S.arrows c₃ c₄) g
      gq : Eq q (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
      ⊢ Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map.Ar …
    -/
    cases hφ he; rw [gq, ← eq_conj_eqToHom, ← φ.map_comp]
    /-
      case im.intro.intro.intro.intro.intro.intro.refl
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S✝ : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      c d : D
      c✝ d✝ : C
      f : Quiver.Hom c✝ d✝
      hf : Membership.mem (S.arrows c✝ d✝) f
      c₄ : C
      q : Quiver.Hom (φ.obj d✝) (φ.obj c₄)
      hq : Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map …
      g : Quiver.Hom d✝ c₄
      he : Eq (φ.obj d✝) (φ.obj d✝)
      hg : Membership.mem (S.arrows d✝ c₄) g
      gq : Eq q (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
      ⊢ Membership.mem ((fun c d => setOf fun x => CategoryTheory.Subgroupoid.Map.Ar …
    -/
    constructor; exact S.mul hf hg
                 /-
                   🎉 no goals
                 -/


theorem mem_map_iff (hφ : Function.Injective φ.obj) (S : Subgroupoid C) {c d : D} (f : c ⟶ d) :
    f ∈ (map φ hφ S).arrows c d ↔
      ∃ (a b : C) (g : a ⟶ b) (ha : φ.obj a = c) (hb : φ.obj b = d) (_hg : g ∈ S.arrows a b),
        f = eqToHom ha.symm ≫ φ.map g ≫ eqToHom hb :=
  Map.arrows_iff φ hφ S f


theorem galoisConnection_map_comap (hφ : Function.Injective φ.obj) :
    GaloisConnection (map φ hφ) (comap φ) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    ⊢ GaloisConnection (CategoryTheory.Subgroupoid.map φ hφ) (CategoryTheory.Subgr …
  -/
  rintro S T; simp_rw [le_iff]; constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      T : CategoryTheory.Subgroupoid D
      ⊢ (∀ {c d : D}, HasSubset.Subset ((CategoryTheory.Subgroupoid.map φ hφ S).arro …
    -/
  · exact fun h c d f fS => h (Map.Arrows.im f fS)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      T : CategoryTheory.Subgroupoid D
      ⊢ (∀ {c d : C}, HasSubset.Subset (S.arrows c d) ((CategoryTheory.Subgroupoid.c …
    -/
  · rintro h _ _ g ⟨a, gφS⟩
    /-
      case mpr.im
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      S : CategoryTheory.Subgroupoid C
      T : CategoryTheory.Subgroupoid D
      h : ∀ {c d : C}, HasSubset.Subset (S.arrows c d) ((CategoryTheory.Subgroupoid. …
      c d : D
      c✝ d✝ : C
      a : Quiver.Hom c✝ d✝
      gφS : Membership.mem (S.arrows c✝ d✝) a
      ⊢ Membership.mem (T.arrows (φ.obj c✝) (φ.obj d✝)) (φ.map a)
    -/
    exact h gφS
    /-
      🎉 no goals
    -/


theorem map_mono (hφ : Function.Injective φ.obj) (S T : Subgroupoid C) :
    S ≤ T → map φ hφ S ≤ map φ hφ T := fun h => (galoisConnection_map_comap φ hφ).monotone_l h


theorem le_comap_map (hφ : Function.Injective φ.obj) (S : Subgroupoid C) :
    S ≤ comap φ (map φ hφ S) :=
  (galoisConnection_map_comap φ hφ).le_u_l S


theorem map_comap_le (hφ : Function.Injective φ.obj) (T : Subgroupoid D) :
    map φ hφ (comap φ T) ≤ T :=
  (galoisConnection_map_comap φ hφ).l_u_le T


theorem map_le_iff_le_comap (hφ : Function.Injective φ.obj) (S : Subgroupoid C)
    (T : Subgroupoid D) : map φ hφ S ≤ T ↔ S ≤ comap φ T :=
  (galoisConnection_map_comap φ hφ).le_iff_le


theorem mem_map_objs_iff (hφ : Function.Injective φ.obj) (d : D) :
    d ∈ (map φ hφ S).objs ↔ ∃ c ∈ S.objs, φ.obj c = d := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    d : D
    ⊢ Iff (Membership.mem (CategoryTheory.Subgroupoid.map φ hφ S).objs d) (Exists  …
  -/
  dsimp [objs, map]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    d : D
    ⊢ Iff (setOf fun x => CategoryTheory.Subgroupoid.Map.Arrows φ hφ S d d x).None …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      d : D
      ⊢ (setOf fun x => CategoryTheory.Subgroupoid.Map.Arrows φ hφ S d d x).Nonempty …
    -/
  · rintro ⟨f, hf⟩
    /-
      case mp.intro
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      d : D
      f : Quiver.Hom d d
      hf : Membership.mem (setOf fun x => CategoryTheory.Subgroupoid.Map.Arrows φ hφ …
      ⊢ Exists fun c => And (S.arrows c c).Nonempty (Eq (φ.obj c) d)
    -/
    change Map.Arrows φ hφ S d d f at hf; rw [Map.arrows_iff] at hf
    /-
      case mp.intro
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      d : D
      f : Quiver.Hom d d
      hf : Exists fun a => Exists fun b => Exists fun g => Exists fun ha => Exists f …
      ⊢ Exists fun c => And (S.arrows c c).Nonempty (Eq (φ.obj c) d)
    -/
    obtain ⟨c, d, g, ec, ed, eg, gS, eg⟩ := hf
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.refl
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      d✝ : D
      c d : C
      g : Quiver.Hom c d
      ec : Eq (φ.obj c) d✝
      ed : Eq (φ.obj d) d✝
      eg : Membership.mem (S.arrows c d) g
      ⊢ Exists fun c => And (S.arrows c c).Nonempty (Eq (φ.obj c) d✝)
    -/
    exact ⟨c, ⟨mem_objs_of_src S eg, ec⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      d : D
      ⊢ (Exists fun c => And (S.arrows c c).Nonempty (Eq (φ.obj c) d)) → (setOf fun  …
    -/
  · rintro ⟨c, ⟨γ, γS⟩, rfl⟩
    /-
      case mpr.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      D : Type u_1
      inst✝ : CategoryTheory.Groupoid D
      φ : CategoryTheory.Functor C D
      hφ : Function.Injective φ.obj
      c : C
      γ : Quiver.Hom c c
      γS : Membership.mem (S.arrows c c) γ
      ⊢ (setOf fun x => CategoryTheory.Subgroupoid.Map.Arrows φ hφ S (φ.obj c) (φ.ob …
    -/
    exact ⟨φ.map γ, ⟨γ, γS⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_objs_eq (hφ : Function.Injective φ.obj) : (map φ hφ S).objs = φ.obj '' S.objs := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    ⊢ Eq (CategoryTheory.Subgroupoid.map φ hφ S).objs (Set.image φ.obj S.objs)
  -/
  ext x; convert mem_map_objs_iff S φ hφ x
         /-
           🎉 no goals
         -/


/-- The image of a functor injective on objects -/
def im (hφ : Function.Injective φ.obj) :=
  map φ hφ ⊤


theorem mem_im_iff (hφ : Function.Injective φ.obj) {c d : D} (f : c ⟶ d) :
    f ∈ (im φ hφ).arrows c d ↔
      ∃ (a b : C) (g : a ⟶ b) (ha : φ.obj a = c) (hb : φ.obj b = d),
        f = eqToHom ha.symm ≫ φ.map g ≫ eqToHom hb := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    c d : D
    f : Quiver.Hom c d
    ⊢ Iff (Membership.mem ((CategoryTheory.Subgroupoid.im φ hφ).arrows c d) f) (Ex …
  -/
  convert Map.arrows_iff φ hφ ⊤ f; simp only [Top.top, mem_univ, exists_true_left]
                                   /-
                                     🎉 no goals
                                   -/


theorem mem_im_objs_iff (hφ : Function.Injective φ.obj) (d : D) :
    d ∈ (im φ hφ).objs ↔ ∃ c : C, φ.obj c = d := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    d : D
    ⊢ Iff (Membership.mem (CategoryTheory.Subgroupoid.im φ hφ).objs d) (Exists fun …
  -/
  simp only [im, mem_map_objs_iff, mem_top_objs, true_and]
  /-
    🎉 no goals
  -/


theorem obj_surjective_of_im_eq_top (hφ : Function.Injective φ.obj) (hφ' : im φ hφ = ⊤) :
    Function.Surjective φ.obj := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
    ⊢ Function.Surjective φ.obj
  -/
  rintro d
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
    d : D
    ⊢ Exists fun a => Eq (φ.obj a) d
  -/
  rw [← mem_im_objs_iff, hφ']
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Groupoid C
    D : Type u_1
    inst✝ : CategoryTheory.Groupoid D
    φ : CategoryTheory.Functor C D
    hφ : Function.Injective φ.obj
    hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
    d : D
    ⊢ Membership.mem Top.top.objs d
  -/
  apply mem_top_objs
  /-
    🎉 no goals
  -/


theorem isNormal_map (hφ : Function.Injective φ.obj) (hφ' : im φ hφ = ⊤) (Sn : S.IsNormal) :
    (map φ hφ S).IsNormal :=
  { wide := fun d => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d : D
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d d) (Categor …
      -/
      obtain ⟨c, rfl⟩ := obj_surjective_of_im_eq_top φ hφ hφ' d
      /-
        case intro
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows (φ.obj c) (φ. …
      -/
      change Map.Arrows φ hφ S _ _ (𝟙 _); rw [← Functor.map_id]
      /-
        case intro
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        ⊢ CategoryTheory.Subgroupoid.Map.Arrows φ hφ S (φ.obj c) (φ.obj c) (φ.map (Cat …
      -/
      constructor; exact Sn.wide c
                   /-
                     🎉 no goals
                   -/
    conj := fun {d d'} g δ hδ => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d d' : D
        g : Quiver.Hom d d'
        δ : Quiver.Hom d d
        hδ : Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d d) δ
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d' d') (Categ …
      -/
      rw [mem_map_iff] at hδ
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d d' : D
        g : Quiver.Hom d d'
        δ : Quiver.Hom d d
        hδ : Exists fun a => Exists fun b => Exists fun g => Exists fun ha => Exists f …
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d' d') (Categ …
      -/
      obtain ⟨c, c', γ, cd, cd', γS, hγ⟩ := hδ; subst_vars; cases hφ cd'
      /-
        case intro.intro.intro.intro.intro.intro.refl
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d' : D
        c : C
        g : Quiver.Hom (φ.obj c) d'
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d' d') (Categ …
      -/
      have : d' ∈ (im φ hφ).objs := by rw [hφ']; apply mem_top_objs
      /-
        case intro.intro.intro.intro.intro.intro.refl
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d' : D
        c : C
        g : Quiver.Hom (φ.obj c) d'
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        this : Membership.mem (CategoryTheory.Subgroupoid.im φ hφ).objs d'
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d' d') (Categ …
      -/
      rw [mem_im_objs_iff] at this
      /-
        case intro.intro.intro.intro.intro.intro.refl
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        d' : D
        c : C
        g : Quiver.Hom (φ.obj c) d'
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        this : Exists fun c => Eq (φ.obj c) d'
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows d' d') (Categ …
      -/
      obtain ⟨c', rfl⟩ := this
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        g : Quiver.Hom (φ.obj c) (φ.obj c')
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows (φ.obj c') (φ …
      -/
      have : g ∈ (im φ hφ).arrows (φ.obj c) (φ.obj c') := by rw [hφ']; trivial
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        g : Quiver.Hom (φ.obj c) (φ.obj c')
        this : Membership.mem ((CategoryTheory.Subgroupoid.im φ hφ).arrows (φ.obj c) ( …
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows (φ.obj c') (φ …
      -/
      rw [mem_im_iff] at this
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        g : Quiver.Hom (φ.obj c) (φ.obj c')
        this : Exists fun a => Exists fun b => Exists fun g_1 => Exists fun ha => Exis …
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows (φ.obj c') (φ …
      -/
      obtain ⟨b, b', f, hb, hb', _, hf⟩ := this; cases hφ hb; cases hφ hb'
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro.intro.intro.intro.intro.in …
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        hb : Eq (φ.obj c) (φ.obj c)
        hb' : Eq (φ.obj c') (φ.obj c')
        f : Quiver.Hom c c'
        ⊢ Membership.mem ((CategoryTheory.Subgroupoid.map φ hφ S).arrows (φ.obj c') (φ …
      -/
      change Map.Arrows φ hφ S (φ.obj c') (φ.obj c') _
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro.intro.intro.intro.intro.in …
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        hb : Eq (φ.obj c) (φ.obj c)
        hb' : Eq (φ.obj c') (φ.obj c')
        f : Quiver.Hom c c'
        ⊢ CategoryTheory.Subgroupoid.Map.Arrows φ hφ S (φ.obj c') (φ.obj c') (Category …
      -/
      simp only [eqToHom_refl, Category.comp_id, Category.id_comp, inv_eq_inv]
      suffices Map.Arrows φ hφ S (φ.obj c') (φ.obj c') (φ.map <| Groupoid.inv f ≫ γ ≫ f) by
        simp only [inv_eq_inv, Functor.map_comp, Functor.map_inv] at this; exact this
      /-
        case intro.intro.intro.intro.intro.intro.refl.intro.intro.intro.intro.intro.in …
        C : Type u
        inst✝¹ : CategoryTheory.Groupoid C
        S : CategoryTheory.Subgroupoid C
        D : Type u_1
        inst✝ : CategoryTheory.Groupoid D
        φ : CategoryTheory.Functor C D
        hφ : Function.Injective φ.obj
        hφ' : Eq (CategoryTheory.Subgroupoid.im φ hφ) Top.top
        Sn : S.IsNormal
        c : C
        γ : Quiver.Hom c c
        γS : Membership.mem (S.arrows c c) γ
        cd' : Eq (φ.obj c) (φ.obj c)
        c' : C
        hb : Eq (φ.obj c) (φ.obj c)
        hb' : Eq (φ.obj c') (φ.obj c')
        f : Quiver.Hom c c'
        ⊢ CategoryTheory.Subgroupoid.Map.Arrows φ hφ S (φ.obj c') (φ.obj c') (φ.map (C …
      -/
      constructor; apply Sn.conj f γS }
                   /-
                     🎉 no goals
                   -/


/-- A subgroupoid is thin (`CategoryTheory.Subgroupoid.IsThin`) if it has at most one arrow between
any two vertices. -/
abbrev IsThin :=
  Quiver.IsThin S.objs


nonrec theorem isThin_iff : S.IsThin ↔ ∀ c : S.objs, Subsingleton (S.arrows c c) := isThin_iff _


/-- A subgroupoid `IsTotallyDisconnected` if it has only isotropy arrows. -/
nonrec abbrev IsTotallyDisconnected :=
  IsTotallyDisconnected S.objs


theorem isTotallyDisconnected_iff :
    S.IsTotallyDisconnected ↔ ∀ c d, (S.arrows c d).Nonempty → c = d := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    ⊢ Iff S.IsTotallyDisconnected (∀ (c d : C), (S.arrows c d).Nonempty → Eq c d)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      ⊢ S.IsTotallyDisconnected → ∀ (c d : C), (S.arrows c d).Nonempty → Eq c d
    -/
  · rintro h c d ⟨f, fS⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : S.IsTotallyDisconnected
      c d : C
      f : Quiver.Hom c d
      fS : Membership.mem (S.arrows c d) f
      ⊢ Eq c d
    -/
    exact congr_arg Subtype.val <| h ⟨c, mem_objs_of_src S fS⟩ ⟨d, mem_objs_of_tgt S fS⟩ ⟨f, fS⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      ⊢ (∀ (c d : C), (S.arrows c d).Nonempty → Eq c d) → S.IsTotallyDisconnected
    -/
  · rintro h ⟨c, hc⟩ ⟨d, hd⟩ ⟨f, fS⟩
    /-
      case mpr.mk.mk.mk
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : ∀ (c d : C), (S.arrows c d).Nonempty → Eq c d
      c : C
      hc : Membership.mem S.objs c
      d : C
      hd : Membership.mem S.objs d
      f : Quiver.Hom ↑⟨c, hc⟩ ↑⟨d, hd⟩
      fS : Membership.mem (S.arrows ↑⟨c, hc⟩ ↑⟨d, hd⟩) f
      ⊢ Eq ⟨c, hc⟩ ⟨d, hd⟩
    -/
    simp only [Subtype.mk_eq_mk]
    /-
      case mpr.mk.mk.mk
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      S : CategoryTheory.Subgroupoid C
      h : ∀ (c d : C), (S.arrows c d).Nonempty → Eq c d
      c : C
      hc : Membership.mem S.objs c
      d : C
      hd : Membership.mem S.objs d
      f : Quiver.Hom ↑⟨c, hc⟩ ↑⟨d, hd⟩
      fS : Membership.mem (S.arrows ↑⟨c, hc⟩ ↑⟨d, hd⟩) f
      ⊢ Eq c d
    -/
    exact h c d ⟨f, fS⟩
    /-
      🎉 no goals
    -/


/-- The isotropy subgroupoid of `S` -/
def disconnect : Subgroupoid C where
  arrows c d := {f | c = d ∧ f ∈ S.arrows c d}
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              ⊢ ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun f => …
            -/
  inv := by rintro _ _ _ ⟨rfl, h⟩; exact ⟨rfl, S.inv h⟩
                                   /-
                                     🎉 no goals
                                   -/
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              ⊢ ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun f  …
            -/
  mul := by rintro _ _ _ _ ⟨rfl, h⟩ _ ⟨rfl, h'⟩; exact ⟨rfl, S.mul h h'⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


                                               /-
                                                 C : Type u
                                                 inst✝ : CategoryTheory.Groupoid C
                                                 S : CategoryTheory.Subgroupoid C
                                                 ⊢ LE.le S.disconnect S
                                               -/
theorem disconnect_le : S.disconnect ≤ S := by rw [le_iff]; rintro _ _ _ ⟨⟩; assumption
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem disconnect_normal (Sn : S.IsNormal) : S.disconnect.IsNormal :=
  { wide := fun c => ⟨rfl, Sn.wide c⟩
    conj := fun _ _ ⟨_, h'⟩ => ⟨rfl, Sn.conj _ h'⟩ }


@[simp]
theorem mem_disconnect_objs_iff {c : C} : c ∈ S.disconnect.objs ↔ c ∈ S.objs :=
  ⟨fun ⟨γ, _, γS⟩ => ⟨γ, γS⟩, fun ⟨γ, γS⟩ => ⟨γ, rfl, γS⟩⟩


theorem disconnect_objs : S.disconnect.objs = S.objs := Set.ext fun _ ↦ mem_disconnect_objs_iff _


theorem disconnect_isTotallyDisconnected : S.disconnect.IsTotallyDisconnected := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    S : CategoryTheory.Subgroupoid C
    ⊢ S.disconnect.IsTotallyDisconnected
  -/
  rw [isTotallyDisconnected_iff]; exact fun c d ⟨_, h, _⟩ => h
                                  /-
                                    🎉 no goals
                                  -/


/-- The full subgroupoid on a set `D : Set C` -/
def full : Subgroupoid C where
  arrows c d := {_f | c ∈ D ∧ d ∈ D}
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              D : Set C
              ⊢ ∀ {c d : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun _f = …
            -/
                                             /-
                                               🎉 no goals
                                             -/
  inv := by rintro _ _ _ ⟨⟩; constructor <;> assumption
                                             /-
                                               🎉 no goals
                                             -/
            /-
              C : Type u
              inst✝ : CategoryTheory.Groupoid C
              S : CategoryTheory.Subgroupoid C
              D : Set C
              ⊢ ∀ {c d e : C} {p : Quiver.Hom c d}, Membership.mem ((fun c d => setOf fun _f …
            -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  mul := by rintro _ _ _ _ ⟨⟩ _ ⟨⟩; constructor <;> assumption
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem full_objs : (full D).objs = D :=
  Set.ext fun _ => ⟨fun ⟨_, h, _⟩ => h, fun h => ⟨𝟙 _, h, h⟩⟩


@[simp]
theorem mem_full_iff {c d : C} {f : c ⟶ d} : f ∈ (full D).arrows c d ↔ c ∈ D ∧ d ∈ D :=
  Iff.rfl


@[simp]
                                                                    /-
                                                                      C : Type u
                                                                      inst✝ : CategoryTheory.Groupoid C
                                                                      D : Set C
                                                                      c : C
                                                                      ⊢ Iff (Membership.mem (CategoryTheory.Subgroupoid.full D).objs c) (Membership. …
                                                                    -/
theorem mem_full_objs_iff {c : C} : c ∈ (full D).objs ↔ c ∈ D := by rw [full_objs]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem full_empty : full ∅ = (⊥ : Subgroupoid C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    ⊢ Eq (CategoryTheory.Subgroupoid.full EmptyCollection.emptyCollection) Bot.bot
  -/
  ext
  /-
    case arrows.h.h.h
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    x✝² x✝¹ : C
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Iff (Membership.mem ((CategoryTheory.Subgroupoid.full EmptyCollection.emptyC …
  -/
  simp only [Bot.bot, mem_full_iff, mem_empty_iff_false, and_self_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem full_univ : full Set.univ = (⊤ : Subgroupoid C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    ⊢ Eq (CategoryTheory.Subgroupoid.full Set.univ) Top.top
  -/
  ext
  /-
    case arrows.h.h.h
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    x✝² x✝¹ : C
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Iff (Membership.mem ((CategoryTheory.Subgroupoid.full Set.univ).arrows x✝² x …
  -/
  simp only [mem_full_iff, mem_univ, and_self, mem_top]
  /-
    🎉 no goals
  -/


theorem full_mono {D E : Set C} (h : D ≤ E) : full D ≤ full E := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    D E : Set C
    h : LE.le D E
    ⊢ LE.le (CategoryTheory.Subgroupoid.full D) (CategoryTheory.Subgroupoid.full E)
  -/
  rw [le_iff]
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    D E : Set C
    h : LE.le D E
    ⊢ ∀ {c d : C}, HasSubset.Subset ((CategoryTheory.Subgroupoid.full D).arrows c  …
  -/
  rintro c d f
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    D E : Set C
    h : LE.le D E
    c d : C
    f : Quiver.Hom c d
    ⊢ Membership.mem ((CategoryTheory.Subgroupoid.full D).arrows c d) f → Membersh …
  -/
  simp only [mem_full_iff]
  /-
    C : Type u
    inst✝ : CategoryTheory.Groupoid C
    D E : Set C
    h : LE.le D E
    c d : C
    f : Quiver.Hom c d
    ⊢ And (Membership.mem D c) (Membership.mem D d) → And (Membership.mem E c) (Me …
  -/
  exact fun ⟨hc, hd⟩ => ⟨h hc, h hd⟩
  /-
    🎉 no goals
  -/

-- Porting note: using `.1` instead of `↑`

theorem full_arrow_eq_iff {c d : (full D).objs} {f g : c ⟶ d} :
    f = g ↔ (f.1 : c.val ⟶ d.val) = g.1 :=
  Subtype.ext_iff


