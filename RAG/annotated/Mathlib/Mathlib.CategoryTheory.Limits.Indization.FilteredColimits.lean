local notation "𝒢" => Functor.op G ⋙ Functor.op (toOver yoneda (colimit F))


/-- (implementation) Pulling out a colimit out of a hom functor is one half of the key lemma. Note
    that all of the heavy lifting actually happens in `CostructuredArrow.toOverCompYonedaColimit`
    and `yonedaYonedaColimit`. -/
noncomputable def compYonedaColimitIsoColimitCompYoneda :
    𝒢 ⋙ yoneda.obj (colimit H) ≅ colimit (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢) := calc
  𝒢 ⋙ yoneda.obj (colimit H) ≅ 𝒢 ⋙ colimit (H ⋙ yoneda) :=
        isoWhiskerLeft G.op (toOverCompYonedaColimit H)
  _ ≅ 𝒢 ⋙ (H ⋙ yoneda).flip ⋙ colim := isoWhiskerLeft _ (colimitIsoFlipCompColim _)
  _ ≅ (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢).flip ⋙ colim := Iso.refl _
  _ ≅ colimit (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢) := (colimitIsoFlipCompColim _).symm


theorem exists_nonempty_limit_obj_of_colimit [IsFiltered K]
    (h : Nonempty <| limit <| 𝒢 ⋙ yoneda.obj (colimit H)) :
    ∃ k, Nonempty <| limit <| 𝒢 ⋙ yoneda.obj (H.obj k) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    h : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruct …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  obtain ⟨t⟩ := h
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  let t₂ := limMap (compYonedaColimitIsoColimitCompYoneda F G H).hom t
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  let t₃ := (colimitLimitIso (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢).flip).inv t₂
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    t₃ : CategoryTheory.Limits.colimit (CategoryTheory.Limits.limit (H.comp (Categ …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  obtain ⟨k, y, -⟩ := Types.jointly_surjective'.{v, max u v} t₃
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    t₃ : CategoryTheory.Limits.colimit (CategoryTheory.Limits.limit (H.comp (Categ …
    k : K
    y : (CategoryTheory.Limits.limit (H.comp (CategoryTheory.yoneda.comp ((Categor …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  refine ⟨k, ⟨?_⟩⟩
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    t₃ : CategoryTheory.Limits.colimit (CategoryTheory.Limits.limit (H.comp (Categ …
    k : K
    y : (CategoryTheory.Limits.limit (H.comp (CategoryTheory.yoneda.comp ((Categor …
    ⊢ CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow.to …
  -/
  let z := (limitObjIsoLimitCompEvaluation (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢).flip k).hom y
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    t₃ : CategoryTheory.Limits.colimit (CategoryTheory.Limits.limit (H.comp (Categ …
    k : K
    y : (CategoryTheory.Limits.limit (H.comp (CategoryTheory.yoneda.comp ((Categor …
    z : CategoryTheory.Limits.limit ((H.comp (CategoryTheory.yoneda.comp ((Categor …
    ⊢ CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow.to …
  -/
  let y := flipCompEvaluation (H ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj 𝒢) k
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    t : CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow. …
    t₂ : CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (H.comp (Categ …
    t₃ : CategoryTheory.Limits.colimit (CategoryTheory.Limits.limit (H.comp (Categ …
    k : K
    y✝ : (CategoryTheory.Limits.limit (H.comp (CategoryTheory.yoneda.comp ((Catego …
    z : CategoryTheory.Limits.limit ((H.comp (CategoryTheory.yoneda.comp ((Categor …
    y : CategoryTheory.Iso ((H.comp (CategoryTheory.yoneda.comp ((CategoryTheory.w …
    ⊢ CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.CostructuredArrow.to …
  -/
  exact (lim.mapIso y).hom z
  /-
    🎉 no goals
  -/


theorem exists_nonempty_limit_obj_of_isColimit [IsFiltered K] {c : Cocone H} (hc : IsColimit c)
    (T : Over (colimit F)) (hT : c.pt ≅ T)
    (h : Nonempty <| limit <| 𝒢 ⋙ yoneda.obj T) :
    ∃ k, Nonempty <| limit <| 𝒢 ⋙ yoneda.obj (H.obj k) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    c : CategoryTheory.Limits.Cocone H
    hc : CategoryTheory.Limits.IsColimit c
    T : CategoryTheory.Over (CategoryTheory.Limits.colimit F)
    hT : CategoryTheory.Iso c.pt T
    h : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruct …
    ⊢ Exists fun k => Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryT …
  -/
  refine exists_nonempty_limit_obj_of_colimit F G H ?_
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    c : CategoryTheory.Limits.Cocone H
    hc : CategoryTheory.Limits.IsColimit c
    T : CategoryTheory.Over (CategoryTheory.Limits.colimit F)
    hT : CategoryTheory.Iso c.pt T
    h : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruct …
    ⊢ Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costructur …
  -/
  suffices T ≅ colimit H from Nonempty.map (lim.map (whiskerLeft 𝒢 (yoneda.map this.hom))) h
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝⁴ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    K : Type v
    inst✝¹ : CategoryTheory.SmallCategory K
    H : CategoryTheory.Functor K (CategoryTheory.Over (CategoryTheory.Limits.colim …
    inst✝ : CategoryTheory.IsFiltered K
    c : CategoryTheory.Limits.Cocone H
    hc : CategoryTheory.Limits.IsColimit c
    T : CategoryTheory.Over (CategoryTheory.Limits.colimit F)
    hT : CategoryTheory.Iso c.pt T
    h : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruct …
    ⊢ CategoryTheory.Iso T (CategoryTheory.Limits.colimit H)
  -/
  refine hT.symm ≪≫ IsColimit.coconePointUniqueUpToIso hc (colimit.isColimit _)
  /-
    🎉 no goals
  -/


theorem isFiltered [IsFiltered I] (hF : ∀ i, IsIndObject (F.obj i)) :
    IsFiltered (CostructuredArrow yoneda (colimit F)) := by
  -- It suffices to show that for any functor `G : J ⥤ CostructuredArrow yoneda (colimit F)` with
  -- `J` finite there is some `X` such that the set
  -- `lim Hom_{CostructuredArrow yoneda (colimit F)}(G·, X)` is nonempty.
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTheory.y …
  -/
  refine IsFiltered.iff_nonempty_limit.mpr (fun {J _ _} G => ?_)

  -- We begin by remarking that `lim Hom_{Over (colimit F)}(yG·, 𝟙 (colimit F))` is nonempty,
  -- simply because `𝟙 (colimit F)` is the terminal object. Here `y` is the functor
  -- `CostructuredArrow yoneda (colimit F) ⥤ Over (colimit F)` induced by `yoneda`.
  have h₁ : Nonempty (limit (G.op ⋙ (toOver _ _).op ⋙ yoneda.obj (Over.mk (𝟙 (colimit F))))) :=
    ⟨Types.Limit.mk _ (fun j => Over.mkIdTerminal.from _) (by simp)⟩

  -- `𝟙 (colimit F)` is the colimit of the diagram in `Over (colimit F)` given by the arrows of
  -- the form `Fi ⟶ colimit F`. Thus, pulling the colimit out of the hom functor and commuting
  -- the finite limit with the filtered colimit, we obtain
  -- `lim_j Hom_{Over (colimit F)}(yGj, 𝟙 (colimit F)) ≅`
  --   `colim_i lim_j Hom_{Over (colimit F)}(yGj, colimit.ι F i)`, and so we find `i` such that
  -- the limit is non-empty.
  obtain ⟨i, hi⟩ := exists_nonempty_limit_obj_of_isColimit F G _
    (colimit.isColimitToOver F) _ (Iso.refl _) h₁

  -- `F.obj i` is a small filtered colimit of representables, say of the functor `H : K ⥤ C`, so
  -- `𝟙 (F.obj i)` is the colimit of the arrows of the form `yHk ⟶ Fi` in `Over Fi`.
  -- Then `colimit.ι F i` is the colimit of the arrows of the form
  -- `H.obj F ⟶ F.obj i ⟶ colimit F` in `Over (colimit F)`.
  /-
    case intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    J : Type v
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    h₁ : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    i : I
    hi : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruc …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (G.op.comp (CategoryTh …
  -/
  obtain ⟨⟨P⟩⟩ := hF i
  let hc : IsColimit ((Over.map (colimit.ι F i)).mapCocone P.cocone.toOver) :=
    isColimitOfPreserves (Over.map _) (Over.isColimitToOver P.coconeIsColimit)

  -- Again, we pull the colimit out of the hom functor and commute limit and colimit to obtain
  -- `lim_j Hom_{Over (colimit F)}(yGj, colimit.ι F i) ≅`
  --   `colim_k lim_j Hom_{Over (colimit F)}(yGj, yHk)`, and so we find `k` such that the limit
  -- is non-empty.
  obtain ⟨k, hk⟩ : ∃ k, Nonempty (limit (G.op ⋙ (toOver yoneda (colimit F)).op ⋙
      yoneda.obj ((toOver yoneda (colimit F)).obj <|
        (pre P.F yoneda (colimit F)).obj <| (map (colimit.ι F i)).obj <| mk _))) :=
    exists_nonempty_limit_obj_of_isColimit F G _ hc _ (Iso.refl _) hi

  /-
    case intro.mk'.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    J : Type v
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    h₁ : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    i : I
    hi : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruc …
    P : CategoryTheory.Limits.IndObjectPresentation (F.obj i)
    hc : CategoryTheory.Limits.IsColimit ((CategoryTheory.Over.map (CategoryTheory …
    k : P.I
    hk : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (G.op.comp (CategoryTh …
  -/
  have htO : (toOver yoneda (colimit F)).FullyFaithful := .ofFullyFaithful _
  -- Since the inclusion `y : CostructuredArrow yoneda (colimit F) ⥤ Over (colimit F)` is fully
  -- faithful, `lim_j Hom_{Over (colimit F)}(yGj, yHk) ≅`
  --   `lim_j Hom_{CostructuredArrow yoneda (colimit F)}(Gj, Hk)` and so `Hk` is the object we're
  -- looking for.
  /-
    case intro.mk'.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    J : Type v
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    h₁ : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    i : I
    hi : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruc …
    P : CategoryTheory.Limits.IndObjectPresentation (F.obj i)
    hc : CategoryTheory.Limits.IsColimit ((CategoryTheory.Over.map (CategoryTheory …
    k : P.I
    hk : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    htO : (CategoryTheory.CostructuredArrow.toOver CategoryTheory.yoneda (Category …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (G.op.comp (CategoryTh …
  -/
  let q := htO.homNatIsoMaxRight
  /-
    case intro.mk'.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    J : Type v
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    h₁ : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    i : I
    hi : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruc …
    P : CategoryTheory.Limits.IndObjectPresentation (F.obj i)
    hc : CategoryTheory.Limits.IsColimit ((CategoryTheory.Over.map (CategoryTheory …
    k : P.I
    hk : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    htO : (CategoryTheory.CostructuredArrow.toOver CategoryTheory.yoneda (Category …
    q : (X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheor …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (G.op.comp (CategoryTh …
  -/
  obtain ⟨t'⟩ := Nonempty.map (limMap (isoWhiskerLeft G.op (q _)).hom) hk
  /-
    case intro.mk'.intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    inst✝ : CategoryTheory.IsFiltered I
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    J : Type v
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    G : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow CategoryTheory. …
    h₁ : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    i : I
    hi : Nonempty (CategoryTheory.Limits.limit ((G.op.comp (CategoryTheory.Costruc …
    P : CategoryTheory.Limits.IndObjectPresentation (F.obj i)
    hc : CategoryTheory.Limits.IsColimit ((CategoryTheory.Over.map (CategoryTheory …
    k : P.I
    hk : Nonempty (CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.Costruc …
    htO : (CategoryTheory.CostructuredArrow.toOver CategoryTheory.yoneda (Category …
    q : (X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheor …
    t' : CategoryTheory.Limits.limit (G.op.comp ((CategoryTheory.yoneda.obj ((Cate …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (G.op.comp (CategoryTh …
  -/
  exact ⟨_, ⟨((preservesLimitIso uliftFunctor.{u, v} _).inv t').down⟩⟩
  /-
    🎉 no goals
  -/


theorem isIndObject_colimit (I : Type v) [SmallCategory I] [IsFiltered I]
    (F : I ⥤ Cᵒᵖ ⥤ Type v) (hF : ∀ i, IsIndObject (F.obj i)) : IsIndObject (colimit F) := by
  have : IsFiltered (CostructuredArrow yoneda (colimit F)) :=
    IndizationClosedUnderFilteredColimitsAux.isFiltered F hF
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.colimit F)
  -/
  refine (isIndObject_iff _).mpr ⟨this, ?_⟩

  -- It remains to show that `CostructuredArrow yoneda (colimit F)` is finally small. Because we
  -- have already shown it is filtered, it suffices to exhibit a small weakly terminal set. For this
  -- we use that all the `CostructuredArrow yoneda (F.obj i)` have small weakly terminal sets.
  have : ∀ i, ∃ (s : Set (CostructuredArrow yoneda (F.obj i))) (_ : Small.{v} s),
      ∀ i, ∃ j ∈ s, Nonempty (i ⟶ j) :=
    fun i => (hF i).finallySmall.exists_small_weakly_terminal_set
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    this : ∀ (i : I), Exists fun s => Exists fun x => ∀ (i_1 : CategoryTheory.Cost …
    ⊢ CategoryTheory.FinallySmall (CategoryTheory.CostructuredArrow CategoryTheory …
  -/
  choose s hs j hjs hj using this
  refine finallySmall_of_small_weakly_terminal_set
    (⋃ i, (map (colimit.ι F i)).obj '' (s i)) (fun A => ?_)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
    s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
    hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
    j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
    hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
    hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
    A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
    ⊢ Exists fun j => And (Membership.mem (Set.iUnion fun i => Set.image (Category …
  -/
  obtain ⟨i, y, hy⟩ := FunctorToTypes.jointly_surjective'.{v, v} F _ (yonedaEquiv A.hom)
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
    s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
    hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
    j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
    hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
    hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
    A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
    i : I
    y : (F.obj i).obj { unop := A.left }
    hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
    ⊢ Exists fun j => And (Membership.mem (Set.iUnion fun i => Set.image (Category …
  -/
  let y' : CostructuredArrow yoneda (F.obj i) := mk (yonedaEquiv.symm y)
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
    s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
    hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
    j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
    hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
    hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
    A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
    i : I
    y : (F.obj i).obj { unop := A.left }
    hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
    y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
    ⊢ Exists fun j => And (Membership.mem (Set.iUnion fun i => Set.image (Category …
  -/
  obtain ⟨x⟩ := hj _ y'
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsFiltered I
    F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
    s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
    hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
    j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
    hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
    hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
    A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
    i : I
    y : (F.obj i).obj { unop := A.left }
    hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
    y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
    x : Quiver.Hom y' (j i y')
    ⊢ Exists fun j => And (Membership.mem (Set.iUnion fun i => Set.image (Category …
  -/
  refine ⟨(map (colimit.ι F i)).obj (j i y'), ?_, ⟨?_⟩⟩
    /-
      case intro.intro.intro.refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type v
      inst✝¹ : CategoryTheory.SmallCategory I
      inst✝ : CategoryTheory.IsFiltered I
      F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
      hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
      this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
      s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
      hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
      j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
      hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
      hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
      A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
      i : I
      y : (F.obj i).obj { unop := A.left }
      hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
      y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
      x : Quiver.Hom y' (j i y')
      ⊢ Membership.mem (Set.iUnion fun i => Set.image (CategoryTheory.CostructuredAr …
    -/
  · simp only [Set.mem_iUnion, Set.mem_image]
    /-
      case intro.intro.intro.refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type v
      inst✝¹ : CategoryTheory.SmallCategory I
      inst✝ : CategoryTheory.IsFiltered I
      F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
      hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
      this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
      s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
      hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
      j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
      hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
      hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
      A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
      i : I
      y : (F.obj i).obj { unop := A.left }
      hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
      y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
      x : Quiver.Hom y' (j i y')
      ⊢ Exists fun i_1 => Exists fun x => And (Membership.mem (s i_1) x) (Eq ((Categ …
    -/
    exact ⟨i, j i y', hjs _ _, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type v
      inst✝¹ : CategoryTheory.SmallCategory I
      inst✝ : CategoryTheory.IsFiltered I
      F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
      hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
      this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
      s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
      hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
      j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
      hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
      hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
      A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
      i : I
      y : (F.obj i).obj { unop := A.left }
      hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
      y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
      x : Quiver.Hom y' (j i y')
      ⊢ Quiver.Hom A ((CategoryTheory.CostructuredArrow.map (CategoryTheory.Limits.c …
    -/
  · refine ?_ ≫ (map (colimit.ι F i)).map x
    /-
      case intro.intro.intro.refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type v
      inst✝¹ : CategoryTheory.SmallCategory I
      inst✝ : CategoryTheory.IsFiltered I
      F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
      hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
      this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
      s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
      hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
      j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
      hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
      hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
      A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
      i : I
      y : (F.obj i).obj { unop := A.left }
      hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
      y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
      x : Quiver.Hom y' (j i y')
      ⊢ Quiver.Hom A ((CategoryTheory.CostructuredArrow.map (CategoryTheory.Limits.c …
    -/
    refine homMk (𝟙 A.left) (yonedaEquiv.injective ?_)
    /-
      case intro.intro.intro.refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type v
      inst✝¹ : CategoryTheory.SmallCategory I
      inst✝ : CategoryTheory.IsFiltered I
      F : CategoryTheory.Functor I (CategoryTheory.Functor (Opposite C) (Type v))
      hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (F.obj i)
      this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryThe …
      s : (i : I) → Set (CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.o …
      hs : ∀ (i : I), Small.{v, max u v} ↑(s i)
      j : (i : I) → CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) …
      hjs : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda  …
      hj : ∀ (i : I) (i_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda ( …
      A : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (CategoryTheory.Lim …
      i : I
      y : (F.obj i).obj { unop := A.left }
      hy : Eq (CategoryTheory.yonedaEquiv A.hom) ((CategoryTheory.Limits.colimit.ι F …
      y' : CategoryTheory.CostructuredArrow CategoryTheory.yoneda (F.obj i) := Categ …
      x : Quiver.Hom y' (j i y')
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
    -/
    simp [-EmbeddingLike.apply_eq_iff_eq, hy, yonedaEquiv_comp, y']
    /-
      🎉 no goals
    -/


