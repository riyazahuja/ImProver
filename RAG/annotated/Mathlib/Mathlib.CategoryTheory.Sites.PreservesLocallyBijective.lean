lemma isLocallyInjective_whisker [H.IsCocontinuous J K] [IsLocallyInjective K f] :
    IsLocallyInjective J (whiskerLeft H.op f) where
  equalizerSieve_mem x y h := H.cover_lift J K (equalizerSieve_mem K f x y h)


lemma isLocallyInjective_of_whisker (hH : CoverPreserving J K H)
    [H.IsCoverDense K] [IsLocallyInjective J (whiskerLeft H.op f)] : IsLocallyInjective K f where
  equalizerSieve_mem {X} a b h := by
    /-
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_7, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
      X : Opposite D
      a b : (CategoryTheory.forget A).obj (F.obj X)
      h : Eq ((f.app X) a) ((f.app X) b)
      ⊢ Membership.mem (K (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    apply K.transitive (H.is_cover_of_isCoverDense K X.unop)
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_7, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
      X : Opposite D
      a b : (CategoryTheory.forget A).obj (F.obj X)
      h : Eq ((f.app X) a) ((f.app X) b)
      ⊢ ∀ ⦃Y : D⦄ ⦃f : Quiver.Hom Y (Opposite.unop X)⦄, (CategoryTheory.Sieve.coverB …
    -/
    intro Y g ⟨⟨Z, lift, map, fac⟩⟩
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_7, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
      X : Opposite D
      a b : (CategoryTheory.forget A).obj (F.obj X)
      h : Eq ((f.app X) a) ((f.app X) b)
      Y : D
      g : Quiver.Hom Y (Opposite.unop X)
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) (Opposite.unop X)
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback g (CategoryTheory.Preshe …
    -/
    rw [← fac, Sieve.pullback_comp]
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_7, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
      X : Opposite D
      a b : (CategoryTheory.forget A).obj (F.obj X)
      h : Eq ((f.app X) a) ((f.app X) b)
      Y : D
      g : Quiver.Hom Y (Opposite.unop X)
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) (Opposite.unop X)
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback lift (CategoryTheory.Sie …
    -/
    apply K.pullback_stable
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_7, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
      X : Opposite D
      a b : (CategoryTheory.forget A).obj (F.obj X)
      h : Eq ((f.app X) a) ((f.app X) b)
      Y : D
      g : Quiver.Hom Y (Opposite.unop X)
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) (Opposite.unop X)
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      ⊢ Membership.mem (K (H.obj Z)) (CategoryTheory.Sieve.pullback map (CategoryThe …
    -/
    refine K.superset_covering (Sieve.functorPullback_pushforward_le H _) ?_
    refine K.superset_covering (Sieve.functorPushforward_monotone H _ ?_)
      (hH.cover_preserve <| equalizerSieve_mem J (whiskerLeft H.op f)
        ((forget A).map (F.map map.op) a) ((forget A).map (F.map map.op) b) ?_)
      /-
        case h.hS.refine_1
        C : Type u_1
        D : Type u_2
        A : Type u_3
        inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
        inst✝³ : CategoryTheory.Category.{u_7, u_3} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        H : CategoryTheory.Functor C D
        F G : CategoryTheory.Functor (Opposite D) A
        f : Quiver.Hom F G
        inst✝² : CategoryTheory.ConcreteCategory A
        hH : CategoryTheory.CoverPreserving J K H
        inst✝¹ : H.IsCoverDense K
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
        X : Opposite D
        a b : (CategoryTheory.forget A).obj (F.obj X)
        h : Eq ((f.app X) a) ((f.app X) b)
        Y : D
        g : Quiver.Hom Y (Opposite.unop X)
        Z : C
        lift : Quiver.Hom Y (H.obj Z)
        map : Quiver.Hom (H.obj Z) (Opposite.unop X)
        fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
        ⊢ LE.le (CategoryTheory.Presheaf.equalizerSieve ((CategoryTheory.forget A).map …
      -/
    · intro W q hq
      /-
        case h.hS.refine_1
        C : Type u_1
        D : Type u_2
        A : Type u_3
        inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
        inst✝³ : CategoryTheory.Category.{u_7, u_3} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        H : CategoryTheory.Functor C D
        F G : CategoryTheory.Functor (Opposite D) A
        f : Quiver.Hom F G
        inst✝² : CategoryTheory.ConcreteCategory A
        hH : CategoryTheory.CoverPreserving J K H
        inst✝¹ : H.IsCoverDense K
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
        X : Opposite D
        a b : (CategoryTheory.forget A).obj (F.obj X)
        h : Eq ((f.app X) a) ((f.app X) b)
        Y : D
        g : Quiver.Hom Y (Opposite.unop X)
        Z : C
        lift : Quiver.Hom Y (H.obj Z)
        map : Quiver.Hom (H.obj Z) (Opposite.unop X)
        fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
        W : C
        q : Quiver.Hom W Z
        hq : (CategoryTheory.Presheaf.equalizerSieve ((CategoryTheory.forget A).map (F …
        ⊢ (CategoryTheory.Sieve.functorPullback H (CategoryTheory.Sieve.pullback map ( …
      -/
      simpa using hq
      /-
        🎉 no goals
      -/
      /-
        case h.hS.refine_2
        C : Type u_1
        D : Type u_2
        A : Type u_3
        inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
        inst✝³ : CategoryTheory.Category.{u_7, u_3} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        H : CategoryTheory.Functor C D
        F G : CategoryTheory.Functor (Opposite D) A
        f : Quiver.Hom F G
        inst✝² : CategoryTheory.ConcreteCategory A
        hH : CategoryTheory.CoverPreserving J K H
        inst✝¹ : H.IsCoverDense K
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
        X : Opposite D
        a b : (CategoryTheory.forget A).obj (F.obj X)
        h : Eq ((f.app X) a) ((f.app X) b)
        Y : D
        g : Quiver.Hom Y (Opposite.unop X)
        Z : C
        lift : Quiver.Hom Y (H.obj Z)
        map : Quiver.Hom (H.obj Z) (Opposite.unop X)
        fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
        ⊢ Eq (((CategoryTheory.whiskerLeft H.op f).app { unop := Z }) ((CategoryTheory …
      -/
    · simp only [comp_obj, op_obj, whiskerLeft_app, Opposite.op_unop]
      /-
        case h.hS.refine_2
        C : Type u_1
        D : Type u_2
        A : Type u_3
        inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
        inst✝³ : CategoryTheory.Category.{u_7, u_3} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        H : CategoryTheory.Functor C D
        F G : CategoryTheory.Functor (Opposite D) A
        f : Quiver.Hom F G
        inst✝² : CategoryTheory.ConcreteCategory A
        hH : CategoryTheory.CoverPreserving J K H
        inst✝¹ : H.IsCoverDense K
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerLe …
        X : Opposite D
        a b : (CategoryTheory.forget A).obj (F.obj X)
        h : Eq ((f.app X) a) ((f.app X) b)
        Y : D
        g : Quiver.Hom Y (Opposite.unop X)
        Z : C
        lift : Quiver.Hom Y (H.obj Z)
        map : Quiver.Hom (H.obj Z) (Opposite.unop X)
        fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
        ⊢ Eq ((f.app { unop := H.obj Z }) ((CategoryTheory.forget A).map (F.map map.op …
      -/
      erw [NatTrans.naturality_apply, NatTrans.naturality_apply, h]
      /-
        🎉 no goals
      -/


lemma isLocallyInjective_whisker_iff (hH : CoverPreserving J K H) [H.IsCocontinuous J K]
    [H.IsCoverDense K] : IsLocallyInjective J (whiskerLeft H.op f) ↔ IsLocallyInjective K f :=
  ⟨fun _ ↦ isLocallyInjective_of_whisker J K H f hH,
    fun _ ↦ isLocallyInjective_whisker J K H f⟩


lemma isLocallySurjective_whisker [H.IsCocontinuous J K] [IsLocallySurjective K f] :
    IsLocallySurjective J (whiskerLeft H.op f) where
  imageSieve_mem a := H.cover_lift J K (imageSieve_mem K f a)


lemma isLocallySurjective_of_whisker (hH : CoverPreserving J K H)
    [H.IsCoverDense K] [IsLocallySurjective J (whiskerLeft H.op f)] : IsLocallySurjective K f where
  imageSieve_mem {X} a := by
    /-
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      ⊢ Membership.mem (K X) (CategoryTheory.Presheaf.imageSieve f a)
    -/
    apply K.transitive (H.is_cover_of_isCoverDense K X)
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      ⊢ ∀ ⦃Y : D⦄ ⦃f_1 : Quiver.Hom Y X⦄, (CategoryTheory.Sieve.coverByImage H X).ar …
    -/
    intro Y g ⟨⟨Z, lift, map, fac⟩⟩
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback g (CategoryTheory.Preshe …
    -/
    rw [← fac, Sieve.pullback_comp]
    /-
      case h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback lift (CategoryTheory.Sie …
    -/
    apply K.pullback_stable
    have hh := hH.cover_preserve <|
      imageSieve_mem J (whiskerLeft H.op f) ((forget A).map (G.map map.op) a)
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      hh : Membership.mem (K (H.obj (Opposite.unop { unop := Z }))) (CategoryTheory. …
      ⊢ Membership.mem (K (H.obj Z)) (CategoryTheory.Sieve.pullback map (CategoryThe …
    -/
    refine K.superset_covering (Sieve.functorPullback_pushforward_le H _) ?_
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      hh : Membership.mem (K (H.obj (Opposite.unop { unop := Z }))) (CategoryTheory. …
      ⊢ Membership.mem (K (H.obj Z)) (CategoryTheory.Sieve.functorPushforward H (Cat …
    -/
    refine K.superset_covering (Sieve.functorPushforward_monotone H _ ?_) hh
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      hh : Membership.mem (K (H.obj (Opposite.unop { unop := Z }))) (CategoryTheory. …
      ⊢ LE.le (CategoryTheory.Presheaf.imageSieve (CategoryTheory.whiskerLeft H.op f …
    -/
    intro W q ⟨x, h⟩
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      hh : Membership.mem (K (H.obj (Opposite.unop { unop := Z }))) (CategoryTheory. …
      W : C
      q : Quiver.Hom W Z
      x : (CategoryTheory.forget A).obj ((H.op.comp F).obj { unop := W })
      h : Eq (((CategoryTheory.whiskerLeft H.op f).app { unop := W }) x) (((H.op.com …
      ⊢ (CategoryTheory.Sieve.functorPullback H (CategoryTheory.Sieve.pullback map ( …
    -/
    simp only [Sieve.functorPullback_apply, Presieve.functorPullback_mem, Sieve.pullback_apply]
    /-
      case h.hS
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      H : CategoryTheory.Functor C D
      F G : CategoryTheory.Functor (Opposite D) A
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.ConcreteCategory A
      hH : CategoryTheory.CoverPreserving J K H
      inst✝¹ : H.IsCoverDense K
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J (CategoryTheory.whiskerL …
      X : D
      a : (CategoryTheory.forget A).obj (G.obj { unop := X })
      Y : D
      g : Quiver.Hom Y X
      Z : C
      lift : Quiver.Hom Y (H.obj Z)
      map : Quiver.Hom (H.obj Z) X
      fac : Eq (CategoryTheory.CategoryStruct.comp lift map) g
      hh : Membership.mem (K (H.obj (Opposite.unop { unop := Z }))) (CategoryTheory. …
      W : C
      q : Quiver.Hom W Z
      x : (CategoryTheory.forget A).obj ((H.op.comp F).obj { unop := W })
      h : Eq (((CategoryTheory.whiskerLeft H.op f).app { unop := W }) x) (((H.op.com …
      ⊢ (CategoryTheory.Presheaf.imageSieve f a).arrows (CategoryTheory.CategoryStru …
    -/
    exact ⟨x, by simpa using h⟩
    /-
      🎉 no goals
    -/


lemma isLocallySurjective_whisker_iff (hH : CoverPreserving J K H) [H.IsCocontinuous J K]
    [H.IsCoverDense K] : IsLocallySurjective J (whiskerLeft H.op f) ↔ IsLocallySurjective K f :=
  ⟨fun _ ↦ isLocallySurjective_of_whisker J K H f hH,
    fun _ ↦ isLocallySurjective_whisker J K H f⟩


