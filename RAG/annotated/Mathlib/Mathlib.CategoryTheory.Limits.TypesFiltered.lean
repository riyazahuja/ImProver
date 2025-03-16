/-- An alternative relation on `Σ j, F.obj j`,
which generates the same equivalence relation as we use to define the colimit in `Type` above,
but that is more convenient when working with filtered colimits.

Elements in `F.obj j` and `F.obj j'` are equivalent if there is some `k : J` to the right
where their images are equal.
-/
protected def Rel (x y : Σ j, F.obj j) : Prop :=
  ∃ (k : _) (f : x.1 ⟶ k) (g : y.1 ⟶ k), F.map f x.2 = F.map g y.2


theorem rel_of_quot_rel (x y : Σ j, F.obj j) :
    Quot.Rel F x y → FilteredColimit.Rel.{v, u} F x y :=
                                   /-
                                     J : Type v
                                     inst✝ : CategoryTheory.Category.{w, v} J
                                     F : CategoryTheory.Functor J (Type u)
                                     x y : Sigma fun j => F.obj j
                                     x✝ : CategoryTheory.Limits.Types.Quot.Rel F x y
                                     f : Quiver.Hom x.fst y.fst
                                     h : Eq y.snd (F.map f x.snd)
                                     ⊢ Eq (F.map f x.snd) (F.map (CategoryTheory.CategoryStruct.id y.fst) y.snd)
                                   -/
  fun ⟨f, h⟩ => ⟨y.1, f, 𝟙 y.1, by rw [← h, FunctorToTypes.map_id_apply]⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem eqvGen_quot_rel_of_rel (x y : Σ j, F.obj j) :
    FilteredColimit.Rel.{v, u} F x y → Relation.EqvGen (Quot.Rel F) x y := fun ⟨k, f, g, h⟩ => by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    x y : Sigma fun j => F.obj j
    x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    h : Eq (F.map f x.snd) (F.map g y.snd)
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) x y
  -/
  refine Relation.EqvGen.trans _ ⟨k, F.map f x.2⟩ _ ?_ ?_
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      x y : Sigma fun j => F.obj j
      x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
      k : J
      f : Quiver.Hom x.fst k
      g : Quiver.Hom y.fst k
      h : Eq (F.map f x.snd) (F.map g y.snd)
      ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) x ⟨k, F.map f x.snd⟩
    -/
  · exact (Relation.EqvGen.rel _ _ ⟨f, rfl⟩)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      x y : Sigma fun j => F.obj j
      x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
      k : J
      f : Quiver.Hom x.fst k
      g : Quiver.Hom y.fst k
      h : Eq (F.map f x.snd) (F.map g y.snd)
      ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) ⟨k, F.map f x.snd⟩ y
    -/
  · exact (Relation.EqvGen.symm _ _ (Relation.EqvGen.rel _ _ ⟨g, h⟩))
    /-
      🎉 no goals
    -/


/-- Recognizing filtered colimits of types. -/
noncomputable def isColimitOf (t : Cocone F) (hsurj : ∀ x : t.pt, ∃ i xi, x = t.ι.app i xi)
    (hinj :
      ∀ i j xi xj,
        t.ι.app i xi = t.ι.app j xj → ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f xi = F.map g xj) :
    IsColimit t := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    hsurj : ∀ (x : t.pt), Exists fun i => Exists fun xi => Eq x (t.ι.app i xi)
    hinj : ∀ (i j : J) (xi : F.obj i) (xj : F.obj j), Eq (t.ι.app i xi) (t.ι.app j …
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  let α : t.pt → J := fun x => (hsurj x).choose
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    hsurj : ∀ (x : t.pt), Exists fun i => Exists fun xi => Eq x (t.ι.app i xi)
    hinj : ∀ (i j : J) (xi : F.obj i) (xj : F.obj j), Eq (t.ι.app i xi) (t.ι.app j …
    α : t.pt → J := fun x => ⋯.choose
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  let f : ∀ (x : t.pt), F.obj (α x) := fun x => (hsurj x).choose_spec.choose
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    hsurj : ∀ (x : t.pt), Exists fun i => Exists fun xi => Eq x (t.ι.app i xi)
    hinj : ∀ (i j : J) (xi : F.obj i) (xj : F.obj j), Eq (t.ι.app i xi) (t.ι.app j …
    α : t.pt → J := fun x => ⋯.choose
    f : (x : t.pt) → F.obj (α x) := fun x => ⋯.choose
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  have hf : ∀ (x : t.pt), x = t.ι.app _ (f x) := fun x => (hsurj x).choose_spec.choose_spec
  exact
    { desc := fun s x => s.ι.app _ (f x)
      fac := fun s j => by
        ext y
        obtain ⟨k, l, g, eq⟩ := hinj _ _ _ _ (hf (t.ι.app j y))
        have h := congr_fun (s.ι.naturality g) (f (t.ι.app j y))
        have h' := congr_fun (s.ι.naturality l) y
        dsimp at h h' ⊢
        rw [← h, ← eq, h']
      uniq := fun s m hm => by
        ext x
        dsimp
        nth_rw 1 [hf x]
        rw [← hm, types_comp_apply] }


protected theorem rel_equiv : _root_.Equivalence (FilteredColimit.Rel.{v, u} F) where
  refl x := ⟨x.1, 𝟙 x.1, 𝟙 x.1, rfl⟩
  symm := fun ⟨k, f, g, h⟩ => ⟨k, g, f, h.symm⟩
  trans {x y z} := fun ⟨k, f, g, h⟩ ⟨k', f', g', h'⟩ =>
    let ⟨l, fl, gl, _⟩ := IsFilteredOrEmpty.cocone_objs k k'
    let ⟨m, n, hn⟩ := IsFilteredOrEmpty.cocone_maps (g ≫ fl) (f' ≫ gl)
    ⟨m, f ≫ fl ≫ n, g' ≫ gl ≫ n,
      calc
                                                                    /-
                                                                      J : Type v
                                                                      inst✝¹ : CategoryTheory.Category.{w, v} J
                                                                      F : CategoryTheory.Functor J (Type u)
                                                                      inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                                                      x y z : Sigma fun j => F.obj j
                                                                      x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                                                      x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                                                      k : J
                                                                      f : Quiver.Hom x.fst k
                                                                      g : Quiver.Hom y.fst k
                                                                      h : Eq (F.map f x.snd) (F.map g y.snd)
                                                                      k' : J
                                                                      f' : Quiver.Hom y.fst k'
                                                                      g' : Quiver.Hom z.fst k'
                                                                      h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                                                      l : J
                                                                      fl : Quiver.Hom k l
                                                                      gl : Quiver.Hom k' l
                                                                      h✝ : True
                                                                      m : J
                                                                      n : Quiver.Hom l m
                                                                      hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStru …
                                                                    -/
        F.map (f ≫ fl ≫ n) x.2 = F.map (fl ≫ n) (F.map f x.2) := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                               /-
                                                 J : Type v
                                                 inst✝¹ : CategoryTheory.Category.{w, v} J
                                                 F : CategoryTheory.Functor J (Type u)
                                                 inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                                 x y z : Sigma fun j => F.obj j
                                                 x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                                 x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                                 k : J
                                                 f : Quiver.Hom x.fst k
                                                 g : Quiver.Hom y.fst k
                                                 h : Eq (F.map f x.snd) (F.map g y.snd)
                                                 k' : J
                                                 f' : Quiver.Hom y.fst k'
                                                 g' : Quiver.Hom z.fst k'
                                                 h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                                 l : J
                                                 fl : Quiver.Hom k l
                                                 gl : Quiver.Hom k' l
                                                 h✝ : True
                                                 m : J
                                                 n : Quiver.Hom l m
                                                 hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                                 ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp fl n) (F.map f x.snd)) (F.map  …
                                               -/
        _ = F.map (fl ≫ n) (F.map g y.2) := by rw [h]
                                               /-
                                                 🎉 no goals
                                               -/
                                           /-
                                             J : Type v
                                             inst✝¹ : CategoryTheory.Category.{w, v} J
                                             F : CategoryTheory.Functor J (Type u)
                                             inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                             x y z : Sigma fun j => F.obj j
                                             x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                             x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                             k : J
                                             f : Quiver.Hom x.fst k
                                             g : Quiver.Hom y.fst k
                                             h : Eq (F.map f x.snd) (F.map g y.snd)
                                             k' : J
                                             f' : Quiver.Hom y.fst k'
                                             g' : Quiver.Hom z.fst k'
                                             h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                             l : J
                                             fl : Quiver.Hom k l
                                             gl : Quiver.Hom k' l
                                             h✝ : True
                                             m : J
                                             n : Quiver.Hom l m
                                             hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                             ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp fl n) (F.map g y.snd)) (F.map  …
                                           -/
        _ = F.map ((g ≫ fl) ≫ n) y.2 := by simp
                                           /-
                                             🎉 no goals
                                           -/
                                            /-
                                              J : Type v
                                              inst✝¹ : CategoryTheory.Category.{w, v} J
                                              F : CategoryTheory.Functor J (Type u)
                                              inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                              x y z : Sigma fun j => F.obj j
                                              x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                              x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                              k : J
                                              f : Quiver.Hom x.fst k
                                              g : Quiver.Hom y.fst k
                                              h : Eq (F.map f x.snd) (F.map g y.snd)
                                              k' : J
                                              f' : Quiver.Hom y.fst k'
                                              g' : Quiver.Hom z.fst k'
                                              h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                              l : J
                                              fl : Quiver.Hom k l
                                              gl : Quiver.Hom k' l
                                              h✝ : True
                                              m : J
                                              n : Quiver.Hom l m
                                              hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                              ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
                                            -/
        _ = F.map ((f' ≫ gl) ≫ n) y.2 := by rw [hn]
                                            /-
                                              🎉 no goals
                                            -/
                                                /-
                                                  J : Type v
                                                  inst✝¹ : CategoryTheory.Category.{w, v} J
                                                  F : CategoryTheory.Functor J (Type u)
                                                  inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                                  x y z : Sigma fun j => F.obj j
                                                  x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                                  x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                                  k : J
                                                  f : Quiver.Hom x.fst k
                                                  g : Quiver.Hom y.fst k
                                                  h : Eq (F.map f x.snd) (F.map g y.snd)
                                                  k' : J
                                                  f' : Quiver.Hom y.fst k'
                                                  g' : Quiver.Hom z.fst k'
                                                  h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                                  l : J
                                                  fl : Quiver.Hom k l
                                                  gl : Quiver.Hom k' l
                                                  h✝ : True
                                                  m : J
                                                  n : Quiver.Hom l m
                                                  hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                                  ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
                                                -/
        _ = F.map (gl ≫ n) (F.map f' y.2) := by simp
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  J : Type v
                                                  inst✝¹ : CategoryTheory.Category.{w, v} J
                                                  F : CategoryTheory.Functor J (Type u)
                                                  inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                                  x y z : Sigma fun j => F.obj j
                                                  x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                                  x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                                  k : J
                                                  f : Quiver.Hom x.fst k
                                                  g : Quiver.Hom y.fst k
                                                  h : Eq (F.map f x.snd) (F.map g y.snd)
                                                  k' : J
                                                  f' : Quiver.Hom y.fst k'
                                                  g' : Quiver.Hom z.fst k'
                                                  h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                                  l : J
                                                  fl : Quiver.Hom k l
                                                  gl : Quiver.Hom k' l
                                                  h✝ : True
                                                  m : J
                                                  n : Quiver.Hom l m
                                                  hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                                  ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp gl n) (F.map f' y.snd)) (F.map …
                                                -/
        _ = F.map (gl ≫ n) (F.map g' z.2) := by rw [h']
                                                /-
                                                  🎉 no goals
                                                -/
                                          /-
                                            J : Type v
                                            inst✝¹ : CategoryTheory.Category.{w, v} J
                                            F : CategoryTheory.Functor J (Type u)
                                            inst✝ : CategoryTheory.IsFilteredOrEmpty J
                                            x y z : Sigma fun j => F.obj j
                                            x✝¹ : CategoryTheory.Limits.Types.FilteredColimit.Rel F x y
                                            x✝ : CategoryTheory.Limits.Types.FilteredColimit.Rel F y z
                                            k : J
                                            f : Quiver.Hom x.fst k
                                            g : Quiver.Hom y.fst k
                                            h : Eq (F.map f x.snd) (F.map g y.snd)
                                            k' : J
                                            f' : Quiver.Hom y.fst k'
                                            g' : Quiver.Hom z.fst k'
                                            h' : Eq (F.map f' y.snd) (F.map g' z.snd)
                                            l : J
                                            fl : Quiver.Hom k l
                                            gl : Quiver.Hom k' l
                                            h✝ : True
                                            m : J
                                            n : Quiver.Hom l m
                                            hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                            ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp gl n) (F.map g' z.snd)) (F.map …
                                          -/
        _ = F.map (g' ≫ gl ≫ n) z.2 := by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


protected theorem rel_eq_eqvGen_quot_rel :
    FilteredColimit.Rel.{v, u} F = Relation.EqvGen (Quot.Rel F) := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.IsFilteredOrEmpty J
    ⊢ Eq (CategoryTheory.Limits.Types.FilteredColimit.Rel F) (Relation.EqvGen (Cat …
  -/
  ext ⟨j, x⟩ ⟨j', y⟩
  /-
    case h.mk.h.mk.a
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.IsFilteredOrEmpty J
    j : J
    x : F.obj j
    j' : J
    y : F.obj j'
    ⊢ Iff (CategoryTheory.Limits.Types.FilteredColimit.Rel F ⟨j, x⟩ ⟨j', y⟩) (Rela …
  -/
  constructor
    /-
      case h.mk.h.mk.a.mp
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : CategoryTheory.IsFilteredOrEmpty J
      j : J
      x : F.obj j
      j' : J
      y : F.obj j'
      ⊢ CategoryTheory.Limits.Types.FilteredColimit.Rel F ⟨j, x⟩ ⟨j', y⟩ → Relation. …
    -/
  · apply eqvGen_quot_rel_of_rel
    /-
      🎉 no goals
    -/
    /-
      case h.mk.h.mk.a.mpr
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : CategoryTheory.IsFilteredOrEmpty J
      j : J
      x : F.obj j
      j' : J
      y : F.obj j'
      ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) ⟨j, x⟩ ⟨j', y⟩ → Ca …
    -/
  · rw [← (FilteredColimit.rel_equiv F).eqvGen_iff]
    /-
      case h.mk.h.mk.a.mpr
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : CategoryTheory.IsFilteredOrEmpty J
      j : J
      x : F.obj j
      j' : J
      y : F.obj j'
      ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) ⟨j, x⟩ ⟨j', y⟩ → Re …
    -/
    exact Relation.EqvGen.mono (rel_of_quot_rel F)
    /-
      🎉 no goals
    -/


theorem colimit_eq_iff_aux {i j : J} {xi : F.obj i} {xj : F.obj j} :
    (colimitCocone F).ι.app i xi = (colimitCocone F).ι.app j xj ↔
      FilteredColimit.Rel.{v, u} F ⟨i, xi⟩ ⟨j, xj⟩ := by
  /-
    J : Type v
    inst✝² : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : J
    xi : F.obj i
    xj : F.obj j
    ⊢ Iff (Eq ((CategoryTheory.Limits.Types.colimitCocone F).ι.app i xi) ((Categor …
  -/
  dsimp
  rw [← (equivShrink _).symm.injective.eq_iff, Equiv.symm_apply_apply, Equiv.symm_apply_apply,
    Quot.eq, FilteredColimit.rel_eq_eqvGen_quot_rel]


theorem isColimit_eq_iff {t : Cocone F} (ht : IsColimit t) {i j : J} {xi : F.obj i} {xj : F.obj j} :
    t.ι.app i xi = t.ι.app j xj ↔ ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f xi = F.map g xj := by
  /-
    J : Type v
    inst✝² : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.Cocone F
    ht : CategoryTheory.Limits.IsColimit t
    i j : J
    xi : F.obj i
    xj : F.obj j
    ⊢ Iff (Eq (t.ι.app i xi) (t.ι.app j xj)) (Exists fun k => Exists fun f => Exis …
  -/
  refine Iff.trans ?_ (colimit_eq_iff_aux F)
  /-
    J : Type v
    inst✝² : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.Cocone F
    ht : CategoryTheory.Limits.IsColimit t
    i j : J
    xi : F.obj i
    xj : F.obj j
    ⊢ Iff (Eq (t.ι.app i xi) (t.ι.app j xj)) (Eq ((CategoryTheory.Limits.Types.col …
  -/
  rw [← (IsColimit.coconePointUniqueUpToIso ht (colimitCoconeIsColimit F)).toEquiv.injective.eq_iff]
  /-
    J : Type v
    inst✝² : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.Cocone F
    ht : CategoryTheory.Limits.IsColimit t
    i j : J
    xi : F.obj i
    xj : F.obj j
    ⊢ Iff (Eq ((ht.coconePointUniqueUpToIso (CategoryTheory.Limits.Types.colimitCo …
  -/
  convert Iff.rfl
  · exact (congrFun
      (IsColimit.comp_coconePointUniqueUpToIso_hom ht (colimitCoconeIsColimit F) _) xi).symm
  · exact (congrFun
      (IsColimit.comp_coconePointUniqueUpToIso_hom ht (colimitCoconeIsColimit F) _) xj).symm


theorem colimit_eq_iff {i j : J} {xi : F.obj i} {xj : F.obj j} :
    colimit.ι F i xi = colimit.ι F j xj ↔
      ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f xi = F.map g xj :=
  isColimit_eq_iff _ (colimit.isColimit F)


