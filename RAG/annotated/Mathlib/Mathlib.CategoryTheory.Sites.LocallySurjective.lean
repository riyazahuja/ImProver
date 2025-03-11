/-- Given `f : F ⟶ G`, a morphism between presieves, and `s : G.obj (op U)`, this is the sieve
of `U` consisting of the `i : V ⟶ U` such that `s` restricted along `i` is in the image of `f`. -/
@[simps (config := .lemmasOnly)]
def imageSieve {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : C} (s : G.obj (op U)) : Sieve U where
  arrows V i := ∃ t : F.obj (op V), f.app _ t = G.map i.op s
  downward_closed := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      U : C
      s : (CategoryTheory.forget A).obj (G.obj { unop := U })
      ⊢ ∀ {Y Z : C} {f_1 : Quiver.Hom Y U}, (fun V i => Exists fun t => Eq ((f.app { …
    -/
    rintro V W i ⟨t, ht⟩ j
    /-
      case intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      U : C
      s : (CategoryTheory.forget A).obj (G.obj { unop := U })
      V W : C
      i : Quiver.Hom V U
      t : (CategoryTheory.forget A).obj (F.obj { unop := V })
      ht : Eq ((f.app { unop := V }) t) ((G.map i.op) s)
      j : Quiver.Hom W V
      ⊢ Exists fun t => Eq ((f.app { unop := W }) t) ((G.map (CategoryTheory.Categor …
    -/
    refine ⟨F.map j.op t, ?_⟩
    /-
      case intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      U : C
      s : (CategoryTheory.forget A).obj (G.obj { unop := U })
      V W : C
      i : Quiver.Hom V U
      t : (CategoryTheory.forget A).obj (F.obj { unop := V })
      ht : Eq ((f.app { unop := V }) t) ((G.map i.op) s)
      j : Quiver.Hom W V
      ⊢ Eq ((f.app { unop := W }) ((F.map j.op) t)) ((G.map (CategoryTheory.Category …
    -/
    rw [op_comp, G.map_comp, comp_apply, ← ht, elementwise_of% f.naturality]
    /-
      🎉 no goals
    -/


theorem imageSieve_eq_sieveOfSection {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : C} (s : G.obj (op U)) :
    imageSieve f s = (imagePresheaf (whiskerRight f (forget A))).sieveOfSection s :=
  rfl


theorem imageSieve_whisker_forget {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : C} (s : G.obj (op U)) :
    imageSieve (whiskerRight f (forget A)) s = imageSieve f s :=
  rfl


theorem imageSieve_app {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : C} (s : F.obj (op U)) :
    imageSieve f (f.app _ s) = ⊤ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    U : C
    s : (CategoryTheory.forget A).obj (F.obj { unop := U })
    ⊢ Eq (CategoryTheory.Presheaf.imageSieve f ((f.app { unop := U }) s)) Top.top
  -/
  ext V i
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    U : C
    s : (CategoryTheory.forget A).obj (F.obj { unop := U })
    V : C
    i : Quiver.Hom V U
    ⊢ Iff ((CategoryTheory.Presheaf.imageSieve f ((f.app { unop := U }) s)).arrows …
  -/
  simp only [Sieve.top_apply, iff_true, imageSieve_apply]
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    U : C
    s : (CategoryTheory.forget A).obj (F.obj { unop := U })
    V : C
    i : Quiver.Hom V U
    ⊢ Exists fun t => Eq ((f.app { unop := V }) t) ((G.map i.op) ((f.app { unop := …
  -/
  have := elementwise_of% (f.naturality i.op)
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    U : C
    s : (CategoryTheory.forget A).obj (F.obj { unop := U })
    V : C
    i : Quiver.Hom V U
    this : ∀ (x : (CategoryTheory.forget A).obj (F.obj { unop := U })), Eq ((f.app …
    ⊢ Exists fun t => Eq ((f.app { unop := V }) t) ((G.map i.op) ((f.app { unop := …
  -/
  exact ⟨F.map i.op s, this s⟩
  /-
    🎉 no goals
  -/


/-- If a morphism `g : V ⟶ U.unop` belongs to the sieve `imageSieve f s g`, then
this is choice of a preimage of `G.map g.op s` in `F.obj (op V)`, see
`app_localPreimage`.-/
noncomputable def localPreimage {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : Cᵒᵖ} (s : G.obj U)
    {V : C} (g : V ⟶ U.unop) (hg : imageSieve f s g) :
    F.obj (op V) :=
  hg.choose


@[simp]
lemma app_localPreimage {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) {U : Cᵒᵖ} (s : G.obj U)
    {V : C} (g : V ⟶ U.unop) (hg : imageSieve f s g) :
    f.app _ (localPreimage f s g hg) = G.map g.op s :=
  hg.choose_spec


/-- A morphism of presheaves `f : F ⟶ G` is locally surjective with respect to a grothendieck
topology if every section of `G` is locally in the image of `f`. -/
class IsLocallySurjective {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) : Prop where
  imageSieve_mem {U : C} (s : G.obj (op U)) : imageSieve f s ∈ J U


lemma imageSieve_mem {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) [IsLocallySurjective J f] {U : Cᵒᵖ}
    (s : G.obj U) : imageSieve f s ∈ J U.unop :=
  IsLocallySurjective.imageSieve_mem _


instance {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) [IsLocallySurjective J f] :
    IsLocallySurjective J (whiskerRight f (forget A)) where
  imageSieve_mem s := imageSieve_mem J f s


theorem isLocallySurjective_iff_imagePresheaf_sheafify_eq_top {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) :
    IsLocallySurjective J f ↔ (imagePresheaf (whiskerRight f (forget A))).sheafify J = ⊤ := by
  simp only [Subpresheaf.ext_iff, funext_iff, Set.ext_iff, top_subpresheaf_obj,
    Set.top_eq_univ, Set.mem_univ, iff_true]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J f) (∀ (x : Opposite C) (x …
  -/
  exact ⟨fun H _ => H.imageSieve_mem, fun H => ⟨H _⟩⟩
  /-
    🎉 no goals
  -/


theorem isLocallySurjective_iff_imagePresheaf_sheafify_eq_top' {F G : Cᵒᵖ ⥤ Type w} (f : F ⟶ G) :
    IsLocallySurjective J f ↔ (imagePresheaf f).sheafify J = ⊤ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J f) (Eq (CategoryTheory.Gr …
  -/
  apply isLocallySurjective_iff_imagePresheaf_sheafify_eq_top
  /-
    🎉 no goals
  -/


theorem isLocallySurjective_iff_whisker_forget {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) :
    IsLocallySurjective J f ↔ IsLocallySurjective J (whiskerRight f (forget A)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J f) (CategoryTheory.Preshe …
  -/
  simp only [isLocallySurjective_iff_imagePresheaf_sheafify_eq_top]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    ⊢ Iff (Eq (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J (Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isLocallySurjective_of_surjective {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G)
    (H : ∀ U, Function.Surjective (f.app U)) : IsLocallySurjective J f where
  imageSieve_mem {U} s := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      H : ∀ (U : Opposite C), Function.Surjective ⇑(f.app U)
      U : C
      s : (CategoryTheory.forget A).obj (G.obj { unop := U })
      ⊢ Membership.mem (J U) (CategoryTheory.Presheaf.imageSieve f s)
    -/
    obtain ⟨t, rfl⟩ := H _ s
    /-
      case intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      H : ∀ (U : Opposite C), Function.Surjective ⇑(f.app U)
      U : C
      t : (CategoryTheory.forget A).obj (F.obj { unop := U })
      ⊢ Membership.mem (J U) (CategoryTheory.Presheaf.imageSieve f ((f.app { unop := …
    -/
    rw [imageSieve_app]
    /-
      case intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom F G
      H : ∀ (U : Opposite C), Function.Surjective ⇑(f.app U)
      U : C
      t : (CategoryTheory.forget A).obj (F.obj { unop := U })
      ⊢ Membership.mem (J U) Top.top
    -/
    exact J.top_mem _
    /-
      🎉 no goals
    -/


instance isLocallySurjective_of_iso {F G : Cᵒᵖ ⥤ A} (f : F ⟶ G) [IsIso f] :
    IsLocallySurjective J f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f
  -/
  apply isLocallySurjective_of_surjective
  /-
    case H
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso f
    ⊢ ∀ (U : Opposite C), Function.Surjective ⇑(f.app U)
  -/
  intro U
  /-
    case H
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso f
    U : Opposite C
    ⊢ Function.Surjective ⇑(f.app U)
  -/
  apply Function.Bijective.surjective
  /-
    case H.hf
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso f
    U : Opposite C
    ⊢ Function.Bijective ⇑(f.app U)
  -/
  rw [← isIso_iff_bijective, ← forget_map_eq_coe]
  /-
    case H.hf
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso f
    U : Opposite C
    ⊢ CategoryTheory.IsIso ((CategoryTheory.forget A).map (f.app U))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isLocallySurjective_comp {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallySurjective J f₁] [IsLocallySurjective J f₂] :
    IsLocallySurjective J (f₁ ≫ f₂) where
  imageSieve_mem s := by
    have : (Sieve.bind (imageSieve f₂ s) fun _ _ h => imageSieve f₁ h.choose) ≤
        imageSieve (f₁ ≫ f₂) s := by
      rintro V i ⟨W, i, j, H, ⟨t', ht'⟩, rfl⟩
      refine ⟨t', ?_⟩
      rw [op_comp, F₃.map_comp, NatTrans.comp_app, comp_apply, comp_apply, ht',
        elementwise_of% f₂.naturality, H.choose_spec]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      U✝ : C
      s : (CategoryTheory.forget A).obj (F₃.obj { unop := U✝ })
      this : LE.le (CategoryTheory.Sieve.bind (CategoryTheory.Presheaf.imageSieve f₂ …
      ⊢ Membership.mem (J U✝) (CategoryTheory.Presheaf.imageSieve (CategoryTheory.Ca …
    -/
    apply J.superset_covering this
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      U✝ : C
      s : (CategoryTheory.forget A).obj (F₃.obj { unop := U✝ })
      this : LE.le (CategoryTheory.Sieve.bind (CategoryTheory.Presheaf.imageSieve f₂ …
      ⊢ Membership.mem (J U✝) (CategoryTheory.Sieve.bind (CategoryTheory.Presheaf.im …
    -/
    apply J.bind_covering
      /-
        case hS
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
        U✝ : C
        s : (CategoryTheory.forget A).obj (F₃.obj { unop := U✝ })
        this : LE.le (CategoryTheory.Sieve.bind (CategoryTheory.Presheaf.imageSieve f₂ …
        ⊢ Membership.mem (J U✝) (CategoryTheory.Presheaf.imageSieve f₂ s)
      -/
    · apply imageSieve_mem
      /-
        🎉 no goals
      -/
      /-
        case hR
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
        U✝ : C
        s : (CategoryTheory.forget A).obj (F₃.obj { unop := U✝ })
        this : LE.le (CategoryTheory.Sieve.bind (CategoryTheory.Presheaf.imageSieve f₂ …
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y U✝⦄ (H : (CategoryTheory.Presheaf.imageSieve f₂  …
      -/
    · intros; apply imageSieve_mem
              /-
                🎉 no goals
              -/


lemma isLocallySurjective_of_isLocallySurjective
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallySurjective J (f₁ ≫ f₂)] :
    IsLocallySurjective J f₂ where
  imageSieve_mem {X} x := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Category …
      X : C
      x : (CategoryTheory.forget A).obj (F₃.obj { unop := X })
      ⊢ Membership.mem (J X) (CategoryTheory.Presheaf.imageSieve f₂ x)
    -/
    refine J.superset_covering ?_ (imageSieve_mem J (f₁ ≫ f₂) x)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Category …
      X : C
      x : (CategoryTheory.forget A).obj (F₃.obj { unop := X })
      ⊢ LE.le (CategoryTheory.Presheaf.imageSieve (CategoryTheory.CategoryStruct.com …
    -/
    intro Y g hg
    exact ⟨f₁.app _ (localPreimage (f₁ ≫ f₂) x g hg),
      by simpa using app_localPreimage (f₁ ≫ f₂) x g hg⟩


lemma isLocallySurjective_of_isLocallySurjective_fac
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} {f₁ : F₁ ⟶ F₂} {f₂ : F₂ ⟶ F₃} {f₃ : F₁ ⟶ F₃} (fac : f₁ ≫ f₂ = f₃)
    [IsLocallySurjective J f₃] : IsLocallySurjective J f₂ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    f₃ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₃
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₂
  -/
  subst fac
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Category …
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₂
  -/
  exact isLocallySurjective_of_isLocallySurjective J f₁ f₂
  /-
    🎉 no goals
  -/


lemma isLocallySurjective_iff_of_fac
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} {f₁ : F₁ ⟶ F₂} {f₂ : F₂ ⟶ F₃} {f₃ : F₁ ⟶ F₃} (fac : f₁ ≫ f₂ = f₃)
    [IsLocallySurjective J f₁] :
    IsLocallySurjective J f₃ ↔ IsLocallySurjective J f₂ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    f₃ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J f₃) (CategoryTheory.Presh …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      f₃ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₃ → CategoryTheory.Presheaf.I …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      f₃ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      a✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₃
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₂
    -/
    exact isLocallySurjective_of_isLocallySurjective_fac J fac
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      f₃ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₂ → CategoryTheory.Presheaf.I …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      f₃ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      a✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₃
    -/
    rw [← fac]
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} A
      inst✝¹ : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      f₃ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      a✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma comp_isLocallySurjective_iff
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallySurjective J f₁] :
    IsLocallySurjective J (f₁ ≫ f₂) ↔ IsLocallySurjective J f₂ :=
  isLocallySurjective_iff_of_fac J rfl


variable {J} in
lemma isLocallySurjective_of_le {K : GrothendieckTopology C} (hJK : J ≤ K) {F G : Cᵒᵖ ⥤ A}
    (f : F ⟶ G) (h : IsLocallySurjective J f) : IsLocallySurjective K f where
                         /-
                           C : Type u
                           inst✝² : CategoryTheory.Category.{v, u} C
                           J : CategoryTheory.GrothendieckTopology C
                           A : Type u'
                           inst✝¹ : CategoryTheory.Category.{v', u'} A
                           inst✝ : CategoryTheory.ConcreteCategory A
                           K : CategoryTheory.GrothendieckTopology C
                           hJK : LE.le J K
                           F G : CategoryTheory.Functor (Opposite C) A
                           f : Quiver.Hom F G
                           h : CategoryTheory.Presheaf.IsLocallySurjective J f
                           U✝ : C
                           s : (CategoryTheory.forget A).obj (G.obj { unop := U✝ })
                           ⊢ Membership.mem (K U✝) (CategoryTheory.Presheaf.imageSieve f s)
                         -/
  imageSieve_mem s := by apply hJK; exact h.1 _
                                    /-
                                      🎉 no goals
                                    -/


lemma isLocallyInjective_of_isLocallyInjective_of_isLocallySurjective
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallyInjective J (f₁ ≫ f₂)] [IsLocallySurjective J f₁] :
    IsLocallyInjective J f₂ where
  equalizerSieve_mem {X} x₁ x₂ h := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      X : Opposite C
      x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
      h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    let S := imageSieve f₁ x₁ ⊓ imageSieve f₁ x₂
    have hS : S ∈ J X.unop := by
      apply J.intersection_covering
      all_goals apply imageSieve_mem
    let T : ∀ ⦃Y : C⦄ (f : Y ⟶ X.unop) (_ : S f), Sieve Y := fun Y f hf =>
      equalizerSieve (localPreimage f₁ x₁ f hf.1) (localPreimage f₁ x₂ f hf.2)
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      X : Opposite C
      x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
      h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
      S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
      hS : Membership.mem (J (Opposite.unop X)) S
      T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    refine J.superset_covering ?_ (J.transitive hS (Sieve.bind S.1 T) ?_)
      /-
        case refine_1
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        ⊢ LE.le (CategoryTheory.Sieve.bind S.arrows T) (CategoryTheory.Presheaf.equali …
      -/
    · rintro Y f ⟨Z, a, g, hg, ha, rfl⟩
      /-
        case refine_1.intro.intro.intro.intro.intro
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        Y Z : C
        a : Quiver.Hom Y Z
        g : Quiver.Hom Z (Opposite.unop X)
        hg : S.arrows g
        ha : (T g hg).arrows a
        ⊢ (CategoryTheory.Presheaf.equalizerSieve x₁ x₂).arrows (CategoryTheory.Catego …
      -/
      simpa using congr_arg (f₁.app _) ha
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop X)⦄, S.arrows f → Membership.mem  …
      -/
    · intro Y f hf
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        Y : C
        f : Quiver.Hom Y (Opposite.unop X)
        hf : S.arrows f
        ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve. …
      -/
      apply J.superset_covering (Sieve.le_pullback_bind _ _ _ hf)
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        Y : C
        f : Quiver.Hom Y (Opposite.unop X)
        hf : S.arrows f
        ⊢ Membership.mem (J Y) (T f hf)
      -/
      apply equalizerSieve_mem J (f₁ ≫ f₂)
      /-
        case refine_2.h
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.Category …
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
        X : Opposite C
        x₁ x₂ : (CategoryTheory.forget A).obj (F₂.obj X)
        h : Eq ((f₂.app X) x₁) ((f₂.app X) x₂)
        S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
        hS : Membership.mem (J (Opposite.unop X)) S
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y (Opposite.unop X)) → S.arrows f → CategoryTheo …
        Y : C
        f : Quiver.Hom Y (Opposite.unop X)
        hf : S.arrows f
        ⊢ Eq (((CategoryTheory.CategoryStruct.comp f₁ f₂).app { unop := Y }) (Category …
      -/
      dsimp
      rw [comp_apply, comp_apply, app_localPreimage, app_localPreimage,
        NatTrans.naturality_apply, NatTrans.naturality_apply, h]


lemma isLocallyInjective_of_isLocallyInjective_of_isLocallySurjective_fac
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} {f₁ : F₁ ⟶ F₂} {f₂ : F₂ ⟶ F₃} (f₃ : F₁ ⟶ F₃) (fac : f₁ ≫ f₂ = f₃)
    [IsLocallyInjective J f₃] [IsLocallySurjective J f₁] :
    IsLocallyInjective J f₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    f₃ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₃
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J f₂
  -/
  subst fac
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryS …
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J f₂
  -/
  exact isLocallyInjective_of_isLocallyInjective_of_isLocallySurjective J f₁ f₂
  /-
    🎉 no goals
  -/


lemma isLocallySurjective_of_isLocallySurjective_of_isLocallyInjective
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallySurjective J (f₁ ≫ f₂)] [IsLocallyInjective J f₂] :
    IsLocallySurjective J f₁ where
  imageSieve_mem {X} x := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      X : C
      x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
      ⊢ Membership.mem (J X) (CategoryTheory.Presheaf.imageSieve f₁ x)
    -/
    let S := imageSieve (f₁ ≫ f₂) (f₂.app _ x)
    let T : ∀ ⦃Y : C⦄ (f : Y ⟶ X) (_ : S f), Sieve Y := fun Y f hf =>
      equalizerSieve (f₁.app _ (localPreimage (f₁ ≫ f₂) (f₂.app _ x) f hf)) (F₂.map f.op x)
    refine J.superset_covering ?_ (J.transitive (imageSieve_mem J (f₁ ≫ f₂) (f₂.app _ x))
      (Sieve.bind S.1 T) ?_)
      /-
        case refine_1
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
        X : C
        x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
        S : CategoryTheory.Sieve X := CategoryTheory.Presheaf.imageSieve (CategoryTheo …
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S.arrows f → CategoryTheory.Sieve Y := fu …
        ⊢ LE.le (CategoryTheory.Sieve.bind S.arrows T) (CategoryTheory.Presheaf.imageS …
      -/
    · rintro Y _ ⟨Z, a, g, hg, ha, rfl⟩
      /-
        case refine_1.intro.intro.intro.intro.intro
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
        X : C
        x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
        S : CategoryTheory.Sieve X := CategoryTheory.Presheaf.imageSieve (CategoryTheo …
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S.arrows f → CategoryTheory.Sieve Y := fu …
        Y Z : C
        a : Quiver.Hom Y Z
        g : Quiver.Hom Z (Opposite.unop { unop := X })
        hg : S.arrows g
        ha : (T g hg).arrows a
        ⊢ (CategoryTheory.Presheaf.imageSieve f₁ x).arrows (CategoryTheory.CategoryStr …
      -/
      exact ⟨F₁.map a.op (localPreimage (f₁ ≫ f₂) _ _ hg), by simpa using ha⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
        X : C
        x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
        S : CategoryTheory.Sieve X := CategoryTheory.Presheaf.imageSieve (CategoryTheo …
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S.arrows f → CategoryTheory.Sieve Y := fu …
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop { unop := X })⦄, (CategoryTheory. …
      -/
    · intro Y f hf
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
        X : C
        x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
        S : CategoryTheory.Sieve X := CategoryTheory.Presheaf.imageSieve (CategoryTheo …
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S.arrows f → CategoryTheory.Sieve Y := fu …
        Y : C
        f : Quiver.Hom Y (Opposite.unop { unop := X })
        hf : (CategoryTheory.Presheaf.imageSieve (CategoryTheory.CategoryStruct.comp f …
        ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve. …
      -/
      apply J.superset_covering (Sieve.le_pullback_bind _ _ _ hf)
      /-
        case refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        A : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} A
        inst✝² : CategoryTheory.ConcreteCategory A
        F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
        f₁ : Quiver.Hom F₁ F₂
        f₂ : Quiver.Hom F₂ F₃
        inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Categor …
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
        X : C
        x : (CategoryTheory.forget A).obj (F₂.obj { unop := X })
        S : CategoryTheory.Sieve X := CategoryTheory.Presheaf.imageSieve (CategoryTheo …
        T : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S.arrows f → CategoryTheory.Sieve Y := fu …
        Y : C
        f : Quiver.Hom Y (Opposite.unop { unop := X })
        hf : (CategoryTheory.Presheaf.imageSieve (CategoryTheory.CategoryStruct.comp f …
        ⊢ Membership.mem (J Y) (T f hf)
      -/
      apply equalizerSieve_mem J f₂
      rw [NatTrans.naturality_apply, ← app_localPreimage (f₁ ≫ f₂) _ _ hf,
        NatTrans.comp_app, comp_apply]


lemma isLocallySurjective_of_isLocallySurjective_of_isLocallyInjective_fac
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} {f₁ : F₁ ⟶ F₂} {f₂ : F₂ ⟶ F₃} (f₃ : F₁ ⟶ F₃) (fac : f₁ ≫ f₂ = f₃)
    [IsLocallySurjective J f₃] [IsLocallyInjective J f₂] :
    IsLocallySurjective J f₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    f₃ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallySurjective J f₃
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₁
  -/
  subst fac
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.Category …
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₁
  -/
  exact isLocallySurjective_of_isLocallySurjective_of_isLocallyInjective J f₁ f₂
  /-
    🎉 no goals
  -/


lemma comp_isLocallyInjective_iff
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallyInjective J f₁] [IsLocallySurjective J f₁] :
    IsLocallyInjective J (f₁ ≫ f₂) ↔ IsLocallyInjective J f₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₁
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategorySt …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStru …
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J f₂
    -/
    exact isLocallyInjective_of_isLocallyInjective_of_isLocallySurjective J f₁ f₂
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J f₂ → CategoryTheory.Presheaf.Is …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₁
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma isLocallySurjective_comp_iff
    {F₁ F₂ F₃ : Cᵒᵖ ⥤ A} (f₁ : F₁ ⟶ F₂) (f₂ : F₂ ⟶ F₃)
    [IsLocallyInjective J f₂] [IsLocallySurjective J f₂] :
    IsLocallySurjective J (f₁ ≫ f₂) ↔ IsLocallySurjective J f₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
    f₁ : Quiver.Hom F₁ F₂
    f₂ : Quiver.Hom F₂ F₃
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryS …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      a✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₁
    -/
    exact isLocallySurjective_of_isLocallySurjective_of_isLocallyInjective J f₁ f₂
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J f₁ → CategoryTheory.Presheaf.I …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) A
      f₁ : Quiver.Hom F₁ F₂
      f₂ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f₂
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₂
      a✝ : CategoryTheory.Presheaf.IsLocallySurjective J f₁
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance {F₁ F₂ : Cᵒᵖ ⥤ Type w} (f : F₁ ⟶ F₂) :
    IsLocallySurjective J (toImagePresheafSheafify J f) where
  imageSieve_mem {X} := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F₁ F₂
      X : C
      ⊢ ∀ (s : (CategoryTheory.forget (Type w)).obj ((CategoryTheory.GrothendieckTop …
    -/
    rintro ⟨s, hs⟩
    /-
      case mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F₁ F₂
      X : C
      s : F₂.obj { unop := X }
      hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
      ⊢ Membership.mem (J X) (CategoryTheory.Presheaf.imageSieve (J.toImagePresheafS …
    -/
    refine J.superset_covering ?_ hs
    /-
      case mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F₁ F₂
      X : C
      s : F₂.obj { unop := X }
      hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
      ⊢ LE.le ((CategoryTheory.GrothendieckTopology.imagePresheaf f).sieveOfSection  …
    -/
    rintro Y g ⟨t, ht⟩
    /-
      case mk.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F₁ F₂
      X : C
      s : F₂.obj { unop := X }
      hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
      Y : C
      g : Quiver.Hom Y X
      t : F₁.obj { unop := Y }
      ht : Eq (f.app { unop := Y } t) (F₂.map g.op s)
      ⊢ (CategoryTheory.Presheaf.imageSieve (J.toImagePresheafSheafify f) ⟨s, hs⟩).a …
    -/
    exact ⟨t, Subtype.ext ht⟩
    /-
      🎉 no goals
    -/


/-- The image of `F` in `J.sheafify F` is isomorphic to the sheafification. -/
noncomputable def sheafificationIsoImagePresheaf (F : Cᵒᵖ ⥤ Type max u v) :
    J.sheafify F ≅ ((imagePresheaf (J.toSheafify F)).sheafify J).toPresheaf where
  hom :=
    J.sheafifyLift (toImagePresheafSheafify J _)
      ((isSheaf_iff_isSheaf_of_type J _).mpr <|
        Subpresheaf.sheafify_isSheaf _ <|
          (isSheaf_iff_isSheaf_of_type J _).mp <| GrothendieckTopology.sheafify_isSheaf J _)
  inv := Subpresheaf.ι _
  hom_inv_id :=
                                                      /-
                                                        C : Type u
                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                        J : CategoryTheory.GrothendieckTopology C
                                                        A : Type u'
                                                        inst✝¹ : CategoryTheory.Category.{v', u'} A
                                                        inst✝ : CategoryTheory.ConcreteCategory A
                                                        F : CategoryTheory.Functor (Opposite C) (Type (max u v))
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify F) (CategoryTheory.Cate …
                                                      -/
    J.sheafify_hom_ext _ _ (J.sheafify_isSheaf _) (by simp [toImagePresheafSheafify])
                                                      /-
                                                        🎉 no goals
                                                      -/
  inv_hom_id := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F : CategoryTheory.Functor (Opposite C) (Type (max u v))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
    -/
    rw [← cancel_mono (Subpresheaf.ι _), Category.id_comp, Category.assoc]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F : CategoryTheory.Functor (Opposite C) (Type (max u v))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
    -/
    refine Eq.trans ?_ (Category.comp_id _)
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F : CategoryTheory.Functor (Opposite C) (Type (max u v))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      F : CategoryTheory.Functor (Opposite C) (Type (max u v))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.sheafifyLift (J.toImagePresheafShe …
    -/
    exact J.sheafify_hom_ext _ _ (J.sheafify_isSheaf _) (by simp [toImagePresheafSheafify])
    /-
      🎉 no goals
    -/


instance isLocallySurjective_toPlus (P : Cᵒᵖ ⥤ Type max u v) :
    IsLocallySurjective J (J.toPlus P) where
  imageSieve_mem x := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      x : (CategoryTheory.forget (Type (max u v))).obj ((J.plusObj P).obj { unop :=  …
      ⊢ Membership.mem (J U✝) (CategoryTheory.Presheaf.imageSieve (J.toPlus P) x)
    -/
    obtain ⟨S, x, rfl⟩ := exists_rep x
    /-
      case intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      S : J.Cover U✝
      x : CategoryTheory.Meq P S
      ⊢ Membership.mem (J U✝) (CategoryTheory.Presheaf.imageSieve (J.toPlus P) (Cate …
    -/
    refine J.superset_covering (fun Y f hf => ⟨x.1 ⟨Y, f, hf⟩, ?_⟩) S.2
    /-
      case intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      S : J.Cover U✝
      x : CategoryTheory.Meq P S
      Y : C
      f : Quiver.Hom Y U✝
      hf : (↑S).arrows f
      ⊢ Eq (((J.toPlus P).app { unop := Y }) (↑x { Y := Y, f := f, hf := hf })) (((J …
    -/
    dsimp
    /-
      case intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      S : J.Cover U✝
      x : CategoryTheory.Meq P S
      Y : C
      f : Quiver.Hom Y U✝
      hf : (↑S).arrows f
      ⊢ Eq (((J.toPlus P).app { unop := Y }) (↑x { Y := Y, f := f, hf := hf })) (((J …
    -/
    rw [toPlus_eq_mk, res_mk_eq_mk_pullback, eq_mk_iff_exists]
    /-
      case intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      S : J.Cover U✝
      x : CategoryTheory.Meq P S
      Y : C
      f : Quiver.Hom Y U✝
      hf : (↑S).arrows f
      ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq ((CategoryTheory.Meq.mk …
    -/
    refine ⟨S.pullback f, homOfLE le_top, 𝟙 _, ?_⟩
    /-
      case intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.ConcreteCategory A
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      U✝ : C
      S : J.Cover U✝
      x : CategoryTheory.Meq P S
      Y : C
      f : Quiver.Hom Y U✝
      hf : (↑S).arrows f
      ⊢ Eq ((CategoryTheory.Meq.mk Top.top (↑x { Y := Y, f := f, hf := hf })).refine …
    -/
    ext ⟨Z, g, hg⟩
    simpa using x.2 (Cover.Relation.mk { hf := hf }
        { hf := S.1.downward_closed hf g } { g₁ := g, g₂ := 𝟙 Z })


instance isLocallySurjective_toSheafify (P : Cᵒᵖ ⥤ Type max u v) :
    IsLocallySurjective J (J.toSheafify P) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (J.toSheafify P)
  -/
  dsimp [GrothendieckTopology.toSheafify]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
  -/
  rw [GrothendieckTopology.plusMap_toPlus]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isLocallySurjective_toSheafify' {D : Type*} [Category D]
    [ConcreteCategory.{max u v} D]
    (P : Cᵒᵖ ⥤ D) [HasWeakSheafify J D] [J.HasSheafCompose (forget D)]
    [J.PreservesSheafification (forget D)] :
    IsLocallySurjective J (toSheafify J P) := by
  rw [isLocallySurjective_iff_whisker_forget,
    ← sheafComposeIso_hom_fac, ← toSheafify_plusPlusIsoSheafify_hom]
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} A
    inst✝⁵ : CategoryTheory.ConcreteCategory A
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : CategoryTheory.HasWeakSheafify J D
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝ : J.PreservesSheafification (CategoryTheory.forget D)
    ⊢ CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.CategoryStruct …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `φ : F₁ ⟶ F₂` is a morphism of sheaves, this is an abbreviation for
`Presheaf.IsLocallySurjective J φ.val`. -/
abbrev IsLocallySurjective := Presheaf.IsLocallySurjective J φ.val


lemma isLocallySurjective_sheafToPresheaf_map_iff :
                                                                                               /-
                                                                                                 C : Type u
                                                                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                                                                 J : CategoryTheory.GrothendieckTopology C
                                                                                                 A : Type u'
                                                                                                 inst✝¹ : CategoryTheory.Category.{v', u'} A
                                                                                                 inst✝ : CategoryTheory.ConcreteCategory A
                                                                                                 F₁ F₂ : CategoryTheory.Sheaf J A
                                                                                                 φ : Quiver.Hom F₁ F₂
                                                                                                 ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective J ((CategoryTheory.sheafToP …
                                                                                               -/
    Presheaf.IsLocallySurjective J ((sheafToPresheaf J A).map φ) ↔ IsLocallySurjective φ := by rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


instance isLocallySurjective_comp [IsLocallySurjective φ] [IsLocallySurjective ψ] :
    IsLocallySurjective (φ ≫ ψ) :=
  Presheaf.isLocallySurjective_comp J φ.val ψ.val


instance isLocallySurjective_of_iso [IsIso φ] : IsLocallySurjective φ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Sheaf J A
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    inst✝ : CategoryTheory.IsIso φ
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
  -/
  have : IsIso φ.val := (inferInstance : IsIso ((sheafToPresheaf J A).map φ))
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Sheaf J A
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    inst✝ : CategoryTheory.IsIso φ
    this : CategoryTheory.IsIso φ.val
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {F G : Sheaf J (Type w)} (f : F ⟶ G) :
    IsLocallySurjective (toImageSheaf f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Sheaf J A
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective (CategoryTheory.GrothendieckTopolog …
  -/
  dsimp [toImageSheaf]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.ConcreteCategory A
    F₁ F₂ F₃ : CategoryTheory.Sheaf J A
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective { val := J.toImagePresheafSheafify  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [IsLocallySurjective φ] :
    IsLocallySurjective ((sheafCompose J (forget A)).map φ) :=
  (Presheaf.isLocallySurjective_iff_whisker_forget J φ.val).1 inferInstance


theorem isLocallySurjective_iff_isIso {F G : Sheaf J (Type w)} (f : F ⟶ G) :
    IsLocallySurjective f ↔ IsIso (imageSheafι f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective f) (CategoryTheory.IsIso (Cate …
  -/
  dsimp only [IsLocallySurjective]
  rw [imageSheafι, Presheaf.isLocallySurjective_iff_imagePresheaf_sheafify_eq_top',
    Subpresheaf.eq_top_iff_isIso]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.Subpresheaf.s …
  -/
  exact isIso_iff_of_reflects_iso (f := imageSheafι f) (F := sheafToPresheaf J (Type w))
  /-
    🎉 no goals
  -/


instance epi_of_isLocallySurjective' {F₁ F₂ : Sheaf J (Type w)} (φ : F₁ ⟶ F₂)
    [IsLocallySurjective φ] : Epi φ where
  left_cancellation {Z} f₁ f₂ h := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      ⊢ Eq f₁ f₂
    -/
    ext X x
    apply (Presieve.isSeparated_of_isSheaf J Z.1 ((isSheaf_iff_isSheaf_of_type _ _).1 Z.2) _
      (Presheaf.imageSieve_mem J φ.val x)).ext
    /-
      case h.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop X)⦄, (CategoryTheory.Presheaf.ima …
    -/
    rintro Y f ⟨s : F₁.val.obj (op Y), hs : φ.val.app _ s = F₂.val.map f.op x⟩
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      ⊢ Eq (Z.val.map f.op (f₁.val.app X x)) (Z.val.map f.op (f₂.val.app X x))
    -/
    dsimp
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      ⊢ Eq (Z.val.map f.op (f₁.val.app X x)) (Z.val.map f.op (f₂.val.app X x))
    -/
    have h₁ := congr_fun (f₁.val.naturality f.op) x
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F₂.val.map f.op) (f₁.val.app { un …
      ⊢ Eq (Z.val.map f.op (f₁.val.app X x)) (Z.val.map f.op (f₂.val.app X x))
    -/
    have h₂ := congr_fun (f₂.val.naturality f.op) x
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F₂.val.map f.op) (f₁.val.app { un …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (F₂.val.map f.op) (f₂.val.app { un …
      ⊢ Eq (Z.val.map f.op (f₁.val.app X x)) (Z.val.map f.op (f₂.val.app X x))
    -/
    dsimp at h₁ h₂
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      h₁ : Eq (f₁.val.app { unop := Y } (F₂.val.map f.op x)) (Z.val.map f.op (f₁.val …
      h₂ : Eq (f₂.val.app { unop := Y } (F₂.val.map f.op x)) (Z.val.map f.op (f₂.val …
      ⊢ Eq (Z.val.map f.op (f₁.val.app X x)) (Z.val.map f.op (f₂.val.app X x))
    -/
    rw [← h₁, ← h₂, ← hs]
    /-
      case h.w.h.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F₁✝ F₂✝ F₃ : CategoryTheory.Sheaf J A
      φ✝ : Quiver.Hom F₁✝ F₂✝
      ψ : Quiver.Hom F₂✝ F₃
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      F₁ F₂ : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F₁ F₂
      inst✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      Z : CategoryTheory.Sheaf J (Type w)
      f₁ f₂ : Quiver.Hom F₂ Z
      h : Eq (CategoryTheory.CategoryStruct.comp φ f₁) (CategoryTheory.CategoryStruc …
      X : Opposite C
      x : F₂.val.obj X
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      s : F₁.val.obj { unop := Y }
      hs : Eq (φ.val.app { unop := Y } s) (F₂.val.map f.op x)
      h₁ : Eq (f₁.val.app { unop := Y } (F₂.val.map f.op x)) (Z.val.map f.op (f₁.val …
      h₂ : Eq (f₂.val.app { unop := Y } (F₂.val.map f.op x)) (Z.val.map f.op (f₂.val …
      ⊢ Eq (f₁.val.app { unop := Y } (φ.val.app { unop := Y } s)) (f₂.val.app { unop …
    -/
    exact congr_fun (congr_app ((sheafToPresheaf J _).congr_map h) (op Y)) s
    /-
      🎉 no goals
    -/


instance epi_of_isLocallySurjective [IsLocallySurjective φ] : Epi φ :=
  (sheafCompose J (forget A)).epi_of_epi_map inferInstance


lemma isLocallySurjective_iff_epi {F G : Sheaf J (Type w)} (φ : F ⟶ G)
    [HasSheafify J (Type w)] :
    IsLocallySurjective φ ↔ Epi φ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F G : CategoryTheory.Sheaf J (Type w)
    φ : Quiver.Hom F G
    inst✝ : CategoryTheory.HasSheafify J (Type w)
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective φ) (CategoryTheory.Epi φ)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ → CategoryTheory.Epi φ
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      a✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      ⊢ CategoryTheory.Epi φ
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      ⊢ CategoryTheory.Epi φ → CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      a✝ : CategoryTheory.Epi φ
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have := epi_of_epi_fac (toImageSheaf_ι φ)
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      a✝ : CategoryTheory.Epi φ
      this : CategoryTheory.Epi (CategoryTheory.GrothendieckTopology.imageSheafι φ)
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    rw [isLocallySurjective_iff_isIso φ]
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      φ : Quiver.Hom F G
      inst✝ : CategoryTheory.HasSheafify J (Type w)
      a✝ : CategoryTheory.Epi φ
      this : CategoryTheory.Epi (CategoryTheory.GrothendieckTopology.imageSheafι φ)
      ⊢ CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.imageSheafι φ)
    -/
    apply isIso_of_mono_of_epi
    /-
      🎉 no goals
    -/


/-- Given a morphism `φ : R ⟶ R'` of presheaves of types and `r' : R'.obj X`,
this is the family of elements of `R` defined over the sieve `Presheaf.imageSieve φ r'`
which sends a map in this sieve to an arbitrary choice of a preimage of the
restriction of `r'`. -/
noncomputable def localPreimage :
    FamilyOfElements R (Presheaf.imageSieve φ r').arrows :=
  fun _ f hf => Presheaf.localPreimage φ r' f hf


lemma isAmalgamation_map_localPreimage :
    ((localPreimage φ r').map φ).IsAmalgamation r' :=
  fun _ f hf => (Presheaf.app_localPreimage φ r' f hf).symm


