local notation "forget" => forget TopCat


/-- A choice of limit cone for a functor `F : J ⥤ TopCat`.
Generally you should just use `limit.cone F`, unless you need the actual definition
(which is in terms of `Types.limitCone`).
-/
def limitCone (F : J ⥤ TopCat.{max v u}) : Cone F where
  pt := TopCat.of { u : ∀ j : J, F.obj j | ∀ {i j : J} (f : i ⟶ j), F.map f (u i) = u j }
  π :=
    { app := fun j =>
        { toFun := fun u => u.val j
          -- Porting note: `continuity` from the original mathlib3 proof failed here.
          continuous_toFun := Continuous.comp (continuous_apply _) (continuous_subtype_val) }
      naturality := fun X Y f => by
        -- Automation fails in various ways in this proof. Why?!
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TopCat
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TopCat
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (To …
        -/
        rw [Category.id_comp]
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TopCat
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq { toFun := fun u => ↑u Y, continuous_toFun := ⋯ } (CategoryTheory.Categor …
        -/
        apply ContinuousMap.ext
        /-
          case h
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TopCat
          X Y : J
          f : Quiver.Hom X Y
          ⊢ ∀ (a : Subtype fun x => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (x i …
        -/
        intro a
        /-
          case h
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TopCat
          X Y : J
          f : Quiver.Hom X Y
          a : Subtype fun x => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (x i)) (x …
          ⊢ Eq ({ toFun := fun u => ↑u Y, continuous_toFun := ⋯ } a) ((CategoryTheory.Ca …
        -/
        exact (a.2 f).symm }
        /-
          🎉 no goals
        -/


/-- A choice of limit cone for a functor `F : J ⥤ TopCat` whose topology is defined as an
infimum of topologies infimum.
Generally you should just use `limit.cone F`, unless you need the actual definition
(which is in terms of `Types.limitCone`).
-/
def limitConeInfi (F : J ⥤ TopCat.{max v u}) : Cone F where
  pt :=
    ⟨(Types.limitCone.{v,u} (F ⋙ forget)).pt,
      ⨅ j, (F.obj j).str.induced ((Types.limitCone.{v,u} (F ⋙ forget)).π.app j)⟩
  π :=
    { app := fun j =>
        ⟨(Types.limitCone.{v,u} (F ⋙ forget)).π.app j, continuous_iff_le_induced.mpr (iInf_le _ _)⟩
      naturality := fun _ _ f =>
        ContinuousMap.coe_injective ((Types.limitCone.{v,u} (F ⋙ forget)).π.naturality f) }


/-- The chosen cone `TopCat.limitCone F` for a functor `F : J ⥤ TopCat` is a limit cone.
Generally you should just use `limit.isLimit F`, unless you need the actual definition
(which is in terms of `Types.limitConeIsLimit`).
-/
def limitConeIsLimit (F : J ⥤ TopCat.{max v u}) : IsLimit (limitCone.{v,u} F) where
  lift S :=
    { toFun := fun x =>
        ⟨fun _ => S.π.app _ x, fun f => by
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i✝ j✝ : J
            f : Quiver.Hom i✝ j✝
            ⊢ Eq ((F.map f) ((fun x_1 => (S.π.app x_1) x) i✝)) ((fun x_1 => (S.π.app x_1)  …
          -/
          dsimp
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i✝ j✝ : J
            f : Quiver.Hom i✝ j✝
            ⊢ Eq ((F.map f) ((S.π.app i✝) x)) ((S.π.app j✝) x)
          -/
          rw [← S.w f]
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i✝ j✝ : J
            f : Quiver.Hom i✝ j✝
            ⊢ Eq ((F.map f) ((S.π.app i✝) x)) ((CategoryTheory.CategoryStruct.comp (S.π.ap …
          -/
          rfl⟩
          /-
            🎉 no goals
          -/
      continuous_toFun :=
        Continuous.subtype_mk (continuous_pi fun j => (S.π.app j).2) fun x i j f => by
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i j : J
            f : Quiver.Hom i j
            ⊢ Eq ((F.map f) ((fun x_1 => (S.π.app x_1) x) i)) ((fun x_1 => (S.π.app x_1) x …
          -/
          dsimp
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i j : J
            f : Quiver.Hom i j
            ⊢ Eq ((F.map f) ((S.π.app i) x)) ((S.π.app j) x)
          -/
          rw [← S.w f]
          /-
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            F : CategoryTheory.Functor J TopCat
            S : CategoryTheory.Limits.Cone F
            x : ↑S.pt
            i j : J
            f : Quiver.Hom i j
            ⊢ Eq ((F.map f) ((S.π.app i) x)) ((CategoryTheory.CategoryStruct.comp (S.π.app …
          -/
          rfl }
          /-
            🎉 no goals
          -/
  uniq S m h := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      S : CategoryTheory.Limits.Cone F
      m : Quiver.Hom S.pt (TopCat.limitCone F).pt
      h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((TopCat.limitCone F). …
      ⊢ Eq m ((fun S => { toFun := fun x => ⟨fun x_1 => (S.π.app x_1) x, ⋯⟩, continu …
    -/
    apply ContinuousMap.ext; intros a; apply Subtype.ext; funext j
    /-
      case h.a.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      S : CategoryTheory.Limits.Cone F
      m : Quiver.Hom S.pt (TopCat.limitCone F).pt
      h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((TopCat.limitCone F). …
      a : ↑S.pt
      j : J
      ⊢ Eq (↑(m a) j) (↑(((fun S => { toFun := fun x => ⟨fun x_1 => (S.π.app x_1) x, …
    -/
    dsimp
    /-
      case h.a.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      S : CategoryTheory.Limits.Cone F
      m : Quiver.Hom S.pt (TopCat.limitCone F).pt
      h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((TopCat.limitCone F). …
      a : ↑S.pt
      j : J
      ⊢ Eq (↑(m a) j) ((S.π.app j) a)
    -/
    rw [← h]
    /-
      case h.a.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      S : CategoryTheory.Limits.Cone F
      m : Quiver.Hom S.pt (TopCat.limitCone F).pt
      h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((TopCat.limitCone F). …
      a : ↑S.pt
      j : J
      ⊢ Eq (↑(m a) j) ((CategoryTheory.CategoryStruct.comp m ((TopCat.limitCone F).π …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The chosen cone `TopCat.limitConeInfi F` for a functor `F : J ⥤ TopCat` is a limit cone.
Generally you should just use `limit.isLimit F`, unless you need the actual definition
(which is in terms of `Types.limitConeIsLimit`).
-/
def limitConeInfiIsLimit (F : J ⥤ TopCat.{max v u}) : IsLimit (limitConeInfi.{v,u} F) := by
  refine IsLimit.ofFaithful forget (Types.limitConeIsLimit.{v,u} (F ⋙ forget))
    -- Porting note: previously could infer all ?_ except continuity
    (fun s => ⟨fun v => ⟨fun j => (Functor.mapCone forget s).π.app j v, ?_⟩, ?_⟩) fun s => ?_
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      v : ↑s.pt
      ⊢ Membership.mem (F.comp (CategoryTheory.forget TopCat)).sections fun j => ((C …
    -/
  · dsimp [Functor.sections]
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      v : ↑s.pt
      ⊢ ∀ {j j' : J} (f : Quiver.Hom j j'), Eq ((CategoryTheory.forget TopCat).map ( …
    -/
    intro _ _ _
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      v : ↑s.pt
      j✝ j'✝ : J
      f✝ : Quiver.Hom j✝ j'✝
      ⊢ Eq ((CategoryTheory.forget TopCat).map (F.map f✝) ((CategoryTheory.forget To …
    -/
    rw [← comp_apply', forget_map_eq_coe, ← s.π.naturality, forget_map_eq_coe]
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      v : ↑s.pt
      j✝ j'✝ : J
      f✝ : Quiver.Hom j✝ j'✝
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).o …
    -/
    dsimp
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      v : ↑s.pt
      j✝ j'✝ : J
      f✝ : Quiver.Hom j✝ j'✝
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id s. …
    -/
    rw [Category.id_comp]
    /-
      🎉 no goals
    -/
  · exact
    continuous_iff_coinduced_le.mpr
      (le_iInf fun j =>
        coinduced_le_iff_le_induced.mp <|
          (continuous_iff_coinduced_le.mp (s.π.app j).continuous : _))
    /-
      case refine_3
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq ((CategoryTheory.forget TopCat).map ((fun s => { toFun := fun v => ⟨fun j …
    -/
  · rfl
    /-
      🎉 no goals
    -/


instance topCat_hasLimitsOfSize : HasLimitsOfSize.{w, v} TopCat.{max v u} where
  has_limits_of_shape _ :=
    { has_limit := fun F =>
        HasLimit.mk
          { cone := limitCone.{v,u} F
            isLimit := limitConeIsLimit F } }


instance topCat_hasLimits : HasLimits TopCat.{u} :=
  TopCat.topCat_hasLimitsOfSize.{u, u}


instance forget_preservesLimitsOfSize :
    PreservesLimitsOfSize.{w, v} (forget : TopCat.{max v u} ⥤ _) where
  preservesLimitsOfShape {_} :=
    { preservesLimit := fun {F} =>
      preservesLimit_of_preserves_limit_cone (limitConeIsLimit.{v,u} F)
          (Types.limitConeIsLimit.{v,u} (F ⋙ forget)) }


instance forget_preservesLimits : PreservesLimits (forget : TopCat.{u} ⥤ _) :=
  TopCat.forget_preservesLimitsOfSize.{u, u}


/-- A choice of colimit cocone for a functor `F : J ⥤ TopCat`.
Generally you should just use `colimit.cocone F`, unless you need the actual definition
(which is in terms of `Types.colimitCocone`).
-/
def colimitCocone (F : J ⥤ TopCat.{max v u}) : Cocone F where
  pt :=
    ⟨(Types.TypeMax.colimitCocone.{v,u} (F ⋙ forget)).pt,
      ⨆ j, (F.obj j).str.coinduced ((Types.TypeMax.colimitCocone (F ⋙ forget)).ι.app j)⟩
  ι :=
    { app := fun j =>
        ⟨(Types.TypeMax.colimitCocone (F ⋙ forget)).ι.app j, continuous_iff_coinduced_le.mpr <|
          -- Porting note: didn't need function before
          le_iSup (fun j =>
            coinduced ((Types.TypeMax.colimitCocone (F ⋙ forget)).ι.app j) (F.obj j).str) j⟩
      naturality := fun _ _ f =>
        ContinuousMap.coe_injective ((Types.TypeMax.colimitCocone (F ⋙ forget)).ι.naturality f) }


/-- The chosen cocone `TopCat.colimitCocone F` for a functor `F : J ⥤ TopCat` is a colimit cocone.
Generally you should just use `colimit.isColimit F`, unless you need the actual definition
(which is in terms of `Types.colimitCoconeIsColimit`).
-/
def colimitCoconeIsColimit (F : J ⥤ TopCat.{max v u}) : IsColimit (colimitCocone F) := by
  refine
    IsColimit.ofFaithful forget (Types.TypeMax.colimitCoconeIsColimit.{v, u} _) (fun s =>
    -- Porting note: it appears notation for forget breaks dot notation (also above)
    -- Porting note: previously function was inferred
      ⟨Quot.lift (fun p => (Functor.mapCocone forget s).ι.app p.fst p.snd) ?_, ?_⟩) fun s => ?_
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      ⊢ ∀ (a b : Sigma fun j => (F.comp (CategoryTheory.forget TopCat)).obj j), Cate …
    -/
  · intro _ _ ⟨_, h⟩
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      a✝ b✝ : Sigma fun j => (F.comp (CategoryTheory.forget TopCat)).obj j
      w✝ : Quiver.Hom a✝.fst b✝.fst
      h : Eq b✝.snd ((F.comp (CategoryTheory.forget TopCat)).map w✝ a✝.snd)
      ⊢ Eq ((fun p => ((CategoryTheory.forget TopCat).mapCocone s).ι.app p.fst p.snd …
    -/
    dsimp
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      a✝ b✝ : Sigma fun j => (F.comp (CategoryTheory.forget TopCat)).obj j
      w✝ : Quiver.Hom a✝.fst b✝.fst
      h : Eq b✝.snd ((F.comp (CategoryTheory.forget TopCat)).map w✝ a✝.snd)
      ⊢ Eq ((CategoryTheory.forget TopCat).map (s.ι.app a✝.fst) a✝.snd) ((CategoryTh …
    -/
    rw [h, Functor.comp_map, ← comp_apply', s.ι.naturality]
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      a✝ b✝ : Sigma fun j => (F.comp (CategoryTheory.forget TopCat)).obj j
      w✝ : Quiver.Hom a✝.fst b✝.fst
      h : Eq b✝.snd ((F.comp (CategoryTheory.forget TopCat)).map w✝ a✝.snd)
      ⊢ Eq ((CategoryTheory.forget TopCat).map (s.ι.app a✝.fst) a✝.snd) ((CategoryTh …
    -/
    dsimp
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      a✝ b✝ : Sigma fun j => (F.comp (CategoryTheory.forget TopCat)).obj j
      w✝ : Quiver.Hom a✝.fst b✝.fst
      h : Eq b✝.snd ((F.comp (CategoryTheory.forget TopCat)).map w✝ a✝.snd)
      ⊢ Eq ((CategoryTheory.forget TopCat).map (s.ι.app a✝.fst) a✝.snd) ((CategoryTh …
    -/
    rw [Category.comp_id]
    /-
      🎉 no goals
    -/
  · exact
    continuous_iff_le_induced.mpr
      (iSup_le fun j =>
        coinduced_le_iff_le_induced.mp <|
          (continuous_iff_coinduced_le.mp (s.ι.app j).continuous : _))
    /-
      case refine_3
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TopCat
      s : CategoryTheory.Limits.Cocone F
      ⊢ Eq ((CategoryTheory.forget TopCat).map ((fun s => { toFun := Quot.lift (fun  …
    -/
  · rfl
    /-
      🎉 no goals
    -/


instance topCat_hasColimitsOfSize : HasColimitsOfSize.{w,v} TopCat.{max v u} where
  has_colimits_of_shape _ :=
    { has_colimit := fun F =>
        HasColimit.mk
          { cocone := colimitCocone F
            isColimit := colimitCoconeIsColimit F } }


instance topCat_hasColimits : HasColimits TopCat.{u} :=
  TopCat.topCat_hasColimitsOfSize.{u, u}


instance forget_preservesColimitsOfSize :
    PreservesColimitsOfSize.{w, v} (forget : TopCat.{max u v} ⥤ _) where
  preservesColimitsOfShape :=
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit F)
          (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget)) }


instance forget_preservesColimits : PreservesColimits (forget : TopCat.{u} ⥤ Type u) :=
  TopCat.forget_preservesColimitsOfSize.{u, u}


/-- The terminal object of `Top` is `PUnit`. -/
def isTerminalPUnit : IsTerminal (TopCat.of PUnit.{u + 1}) :=
  haveI : ∀ X, Unique (X ⟶ TopCat.of PUnit.{u + 1}) := fun X =>
                                                            /-
                                                              J : Type v
                                                              inst✝ : CategoryTheory.Category.{w, v} J
                                                              X : TopCat
                                                              f : Quiver.Hom X (TopCat.of PUnit.{u + 1})
                                                              ⊢ Eq f Inhabited.default
                                                            -/
    ⟨⟨⟨fun _ => PUnit.unit, continuous_const⟩⟩, fun f => by ext; aesop⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  Limits.IsTerminal.ofUnique _


/-- The terminal object of `Top` is `PUnit`. -/
def terminalIsoPUnit : ⊤_ TopCat.{u} ≅ TopCat.of PUnit :=
  terminalIsTerminal.uniqueUpToIso isTerminalPUnit


/-- The initial object of `Top` is `PEmpty`. -/
def isInitialPEmpty : IsInitial (TopCat.of PEmpty.{u + 1}) :=
  haveI : ∀ X, Unique (TopCat.of PEmpty.{u + 1} ⟶ X) := fun X =>
                           /-
                             J : Type v
                             inst✝ : CategoryTheory.Category.{w, v} J
                             X : TopCat
                             ⊢ Continuous fun x => PEmpty.elim x
                           -/
                           /-
                             🎉 no goals
                           -/
    ⟨⟨⟨fun x => x.elim, by continuity⟩⟩, fun f => by ext ⟨⟩⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
  Limits.IsInitial.ofUnique _


/-- The initial object of `Top` is `PEmpty`. -/
def initialIsoPEmpty : ⊥_ TopCat.{u} ≅ TopCat.of PEmpty :=
  initialIsInitial.uniqueUpToIso isInitialPEmpty


