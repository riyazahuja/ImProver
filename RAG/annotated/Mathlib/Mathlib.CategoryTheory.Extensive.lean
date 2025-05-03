/-- A category has pullback of inclusions if it has all pullbacks along coproduct injections. -/
class HasPullbacksOfInclusions (C : Type u) [Category.{v} C] [HasBinaryCoproducts C] : Prop where
  [hasPullbackInl : ∀ {X Y Z : C} (f : Z ⟶ X ⨿ Y), HasPullback coprod.inl f]


/--
A functor preserves pullback of inclusions if it preserves all pullbacks along coproduct injections.
-/
class PreservesPullbacksOfInclusions {C : Type*} [Category C] {D : Type*} [Category D]
    (F : C ⥤ D) [HasBinaryCoproducts C] where
  [preservesPullbackInl : ∀ {X Y Z : C} (f : Z ⟶ X ⨿ Y), PreservesLimit (cospan coprod.inl f) F]


/-- A category is (finitary) pre-extensive if it has finite coproducts,
and binary coproducts are universal. -/
class FinitaryPreExtensive (C : Type u) [Category.{v} C] : Prop where
  [hasFiniteCoproducts : HasFiniteCoproducts C]
  [hasPullbacksOfInclusions : HasPullbacksOfInclusions C]
  /-- In a finitary extensive category, all coproducts are van Kampen -/
  universal' : ∀ {X Y : C} (c : BinaryCofan X Y), IsColimit c → IsUniversalColimit c


/-- A category is (finitary) extensive if it has finite coproducts,
and binary coproducts are van Kampen. -/
class FinitaryExtensive (C : Type u) [Category.{v} C] : Prop where
  [hasFiniteCoproducts : HasFiniteCoproducts C]
  [hasPullbacksOfInclusions : HasPullbacksOfInclusions C]
  /-- In a finitary extensive category, all coproducts are van Kampen -/
  van_kampen' : ∀ {X Y : C} (c : BinaryCofan X Y), IsColimit c → IsVanKampenColimit c


theorem FinitaryExtensive.vanKampen [FinitaryExtensive C] {F : Discrete WalkingPair ⥤ C}
    (c : Cocone F) (hc : IsColimit c) : IsVanKampenColimit c := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  let X := F.obj ⟨WalkingPair.left⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X : C := F.obj { as := CategoryTheory.Limits.WalkingPair.left }
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  let Y := F.obj ⟨WalkingPair.right⟩
  have : F = pair X Y := by
    apply Functor.hext
    · rintro ⟨⟨⟩⟩ <;> rfl
    · rintro ⟨⟨⟩⟩ ⟨j⟩ ⟨⟨rfl : _ = j⟩⟩ <;> simp [X, Y]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X : C := F.obj { as := CategoryTheory.Limits.WalkingPair.left }
    Y : C := F.obj { as := CategoryTheory.Limits.WalkingPair.right }
    this : Eq F (CategoryTheory.Limits.pair X Y)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  clear_value X Y
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    Y X : C
    this : Eq F (CategoryTheory.Limits.pair X Y)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  subst this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    Y X : C
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  exact FinitaryExtensive.van_kampen' c hc
  /-
    🎉 no goals
  -/


instance (priority := 100) [HasBinaryCoproducts C] [HasPullbacks C] :
    HasPullbacksOfInclusions C := ⟨⟩


instance preservesPullbackInl' :
    HasPullback f coprod.inl :=
  hasPullback_symmetry _ _


instance hasPullbackInr' :
    HasPullback f coprod.inr := by
  have : IsPullback (𝟙 _) (f ≫ (coprod.braiding X Y).hom) f (coprod.braiding Y X).hom :=
    IsPullback.of_horiz_isIso ⟨by simp⟩
  /-
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝² : CategoryTheory.Category.{v'', u''} D
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    X Y Z : C
    f : Quiver.Hom Z (CategoryTheory.Limits.coprod X Y)
    this : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id Z) (Categor …
    ⊢ CategoryTheory.Limits.HasPullback f CategoryTheory.Limits.coprod.inr
  -/
  have := (IsPullback.of_hasPullback (f ≫ (coprod.braiding X Y).hom) coprod.inl).paste_horiz this
  simp only [coprod.braiding_hom, Category.comp_id, colimit.ι_desc, BinaryCofan.mk_pt,
    BinaryCofan.ι_app_left, BinaryCofan.mk_inl] at this
  /-
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝² : CategoryTheory.Category.{v'', u''} D
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    X Y Z : C
    f : Quiver.Hom Z (CategoryTheory.Limits.coprod X Y)
    this✝ : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id Z) (Catego …
    this : CategoryTheory.IsPullback (CategoryTheory.Limits.pullback.fst (Category …
    ⊢ CategoryTheory.Limits.HasPullback f CategoryTheory.Limits.coprod.inr
  -/
  exact ⟨⟨⟨_, this.isLimit⟩⟩⟩
  /-
    🎉 no goals
  -/


instance hasPullbackInr :
    HasPullback coprod.inr f :=
  hasPullback_symmetry _ _


noncomputable
instance (priority := 100) [PreservesLimitsOfShape WalkingCospan F] :
    PreservesPullbacksOfInclusions F := ⟨⟩


noncomputable
instance preservesPullbackInl' :
    PreservesLimit (cospan f coprod.inl) F :=
  preservesPullback_symmetry _ _ _


noncomputable
instance preservesPullbackInr' :
    PreservesLimit (cospan f coprod.inr) F := by
  /-
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D✝ : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D✝
    X✝ Y✝ : C
    D : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} D
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.PreservesPullbacksOfInclusions F
    X Y Z : C
    f : Quiver.Hom Z (CategoryTheory.Limits.coprod X Y)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f Categor …
  -/
  apply preservesLimit_of_iso_diagram (K₁ := cospan (f ≫ (coprod.braiding X Y).hom) coprod.inl)
  /-
    case h
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D✝ : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D✝
    X✝ Y✝ : C
    D : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} D
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.PreservesPullbacksOfInclusions F
    X Y Z : C
    f : Quiver.Hom Z (CategoryTheory.Limits.coprod X Y)
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.cospan (CategoryTheory.CategoryStr …
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  apply cospanExt (Iso.refl _) (Iso.refl _) (coprod.braiding X Y).symm <;> simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


noncomputable
instance preservesPullbackInr :
    PreservesLimit (cospan coprod.inr f) F :=
  preservesPullback_symmetry _ _ _


instance (priority := 100) FinitaryExtensive.toFinitaryPreExtensive [FinitaryExtensive C] :
    FinitaryPreExtensive C :=
  ⟨fun c hc ↦ (FinitaryExtensive.van_kampen' c hc).isUniversal⟩


theorem FinitaryExtensive.mono_inr_of_isColimit [FinitaryExtensive C] {c : BinaryCofan X Y}
    (hc : IsColimit c) : Mono c.inr :=
  BinaryCofan.mono_inr_of_isVanKampen (FinitaryExtensive.vanKampen c hc)


theorem FinitaryExtensive.mono_inl_of_isColimit [FinitaryExtensive C] {c : BinaryCofan X Y}
    (hc : IsColimit c) : Mono c.inl :=
  FinitaryExtensive.mono_inr_of_isColimit (BinaryCofan.isColimitFlip hc)


instance [FinitaryExtensive C] (X Y : C) : Mono (coprod.inl : X ⟶ X ⨿ Y) :=
  (FinitaryExtensive.mono_inl_of_isColimit (coprodIsCoprod X Y) : _)


instance [FinitaryExtensive C] (X Y : C) : Mono (coprod.inr : Y ⟶ X ⨿ Y) :=
  (FinitaryExtensive.mono_inr_of_isColimit (coprodIsCoprod X Y) : _)


theorem FinitaryExtensive.isPullback_initial_to_binaryCofan [FinitaryExtensive C]
    {c : BinaryCofan X Y} (hc : IsColimit c) :
    IsPullback (initial.to _) (initial.to _) c.inl c.inr :=
  BinaryCofan.isPullback_initial_to_of_isVanKampen (FinitaryExtensive.vanKampen c hc)


instance (priority := 100) hasStrictInitialObjects_of_finitaryPreExtensive
    [FinitaryPreExtensive C] : HasStrictInitialObjects C :=
  hasStrictInitial_of_isUniversal (FinitaryPreExtensive.universal' _
    ((BinaryCofan.isColimit_iff_isIso_inr initialIsInitial _).mpr (by
      /-
        J : Type v'
        inst✝³ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝¹ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.BinaryCofan.mk (CategoryTheory.C …
      -/
      dsimp
      /-
        J : Type v'
        inst✝³ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝¹ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id (CategoryTheory.Limit …
      -/
      infer_instance)).some)
      /-
        🎉 no goals
      -/


theorem finitaryExtensive_iff_of_isTerminal (C : Type u) [Category.{v} C] [HasFiniteCoproducts C]
    [HasPullbacksOfInclusions C]
    (T : C) (HT : IsTerminal T) (c₀ : BinaryCofan T T) (hc₀ : IsColimit c₀) :
    FinitaryExtensive C ↔ IsVanKampenColimit c₀ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    ⊢ Iff (CategoryTheory.FinitaryExtensive C) (CategoryTheory.IsVanKampenColimit  …
  -/
  refine ⟨fun H => H.van_kampen' c₀ hc₀, fun H => ?_⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    H : CategoryTheory.IsVanKampenColimit c₀
    ⊢ CategoryTheory.FinitaryExtensive C
  -/
  constructor
  /-
    case van_kampen'
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    H : CategoryTheory.IsVanKampenColimit c₀
    ⊢ ∀ {X Y : C} (c : CategoryTheory.Limits.BinaryCofan X Y), CategoryTheory.Limi …
  -/
  simp_rw [BinaryCofan.isVanKampen_iff] at H ⊢
  /-
    case van_kampen'
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
    ⊢ ∀ {X Y : C} (c : CategoryTheory.Limits.BinaryCofan X Y), CategoryTheory.Limi …
  -/
  intro X Y c hc X' Y' c' αX αY f hX hY
  obtain ⟨d, hd, hd'⟩ :=
    Limits.BinaryCofan.IsColimit.desc' hc (HT.from _ ≫ c₀.inl) (HT.from _ ≫ c₀.inr)
  rw [H c' (αX ≫ HT.from _) (αY ≫ HT.from _) (f ≫ d) (by rw [← reassoc_of% hX, hd, Category.assoc])
      (by rw [← reassoc_of% hY, hd', Category.assoc])]
  /-
    case van_kampen'.mk.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    hc : CategoryTheory.Limits.IsColimit c
    X' Y' : C
    c' : CategoryTheory.Limits.BinaryCofan X' Y'
    αX : Quiver.Hom X' X
    αY : Quiver.Hom Y' Y
    f : Quiver.Hom c'.pt c.pt
    hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
    hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
    d : Quiver.Hom c.pt (((CategoryTheory.Functor.const (CategoryTheory.Discrete C …
    hd : Eq (CategoryTheory.CategoryStruct.comp c.inl d) (CategoryTheory.CategoryS …
    hd' : Eq (CategoryTheory.CategoryStruct.comp c.inr d) (CategoryTheory.Category …
    ⊢ Iff (And (CategoryTheory.IsPullback c'.inl (CategoryTheory.CategoryStruct.co …
  -/
  obtain ⟨hl, hr⟩ := (H c (HT.from _) (HT.from _) d hd.symm hd'.symm).mp ⟨hc⟩
  /-
    case van_kampen'.mk.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.HasPullbacksOfInclusions C
    T : C
    HT : CategoryTheory.Limits.IsTerminal T
    c₀ : CategoryTheory.Limits.BinaryCofan T T
    hc₀ : CategoryTheory.Limits.IsColimit c₀
    H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    hc : CategoryTheory.Limits.IsColimit c
    X' Y' : C
    c' : CategoryTheory.Limits.BinaryCofan X' Y'
    αX : Quiver.Hom X' X
    αY : Quiver.Hom Y' Y
    f : Quiver.Hom c'.pt c.pt
    hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
    hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
    d : Quiver.Hom c.pt (((CategoryTheory.Functor.const (CategoryTheory.Discrete C …
    hd : Eq (CategoryTheory.CategoryStruct.comp c.inl d) (CategoryTheory.CategoryS …
    hd' : Eq (CategoryTheory.CategoryStruct.comp c.inr d) (CategoryTheory.Category …
    hl : CategoryTheory.IsPullback c.inl (HT.from X) d c₀.inl
    hr : CategoryTheory.IsPullback c.inr (HT.from Y) d c₀.inr
    ⊢ Iff (And (CategoryTheory.IsPullback c'.inl (CategoryTheory.CategoryStruct.co …
  -/
  rw [hl.paste_vert_iff hX.symm, hr.paste_vert_iff hY.symm]
  /-
    🎉 no goals
  -/


instance types.finitaryExtensive : FinitaryExtensive (Type u) := by
  classical
  rw [finitaryExtensive_iff_of_isTerminal (Type u) PUnit Types.isTerminalPunit _
      (Types.binaryCoproductColimit _ _)]
  apply BinaryCofan.isVanKampen_mk _ _ (fun X Y => Types.binaryCoproductColimit X Y) _
      fun f g => (Limits.Types.pullbackLimitCone f g).2
  · intros _ _ _ _ f hαX hαY
    constructor
    · refine ⟨⟨hαX.symm⟩, ⟨PullbackCone.isLimitAux' _ ?_⟩⟩
      intro s
      have : ∀ x, ∃! y, s.fst x = Sum.inl y := by
        intro x
        cases' h : s.fst x with val val
        · simp only [Types.binaryCoproductCocone_pt, Functor.const_obj_obj, Sum.inl.injEq,
            existsUnique_eq']
        · apply_fun f at h
          cases ((congr_fun s.condition x).symm.trans h).trans (congr_fun hαY val : _).symm
      delta ExistsUnique at this
      choose l hl hl' using this
      exact ⟨l, (funext hl).symm, Types.isTerminalPunit.hom_ext _ _,
        fun {l'} h₁ _ => funext fun x => hl' x (l' x) (congr_fun h₁ x).symm⟩
    · refine ⟨⟨hαY.symm⟩, ⟨PullbackCone.isLimitAux' _ ?_⟩⟩
      intro s
      have : ∀ x, ∃! y, s.fst x = Sum.inr y := by
        intro x
        cases' h : s.fst x with val val
        · apply_fun f at h
          cases ((congr_fun s.condition x).symm.trans h).trans (congr_fun hαX val : _).symm
        · simp only [Types.binaryCoproductCocone_pt, Functor.const_obj_obj, Sum.inr.injEq,
            existsUnique_eq']
      delta ExistsUnique at this
      choose l hl hl' using this
      exact ⟨l, (funext hl).symm, Types.isTerminalPunit.hom_ext _ _,
        fun {l'} h₁ _ => funext fun x => hl' x (l' x) (congr_fun h₁ x).symm⟩
  · intro Z f
    dsimp [Limits.Types.binaryCoproductCocone]
    delta Types.PullbackObj
    have : ∀ x, f x = Sum.inl PUnit.unit ∨ f x = Sum.inr PUnit.unit := by
      intro x
      rcases f x with (⟨⟨⟩⟩ | ⟨⟨⟩⟩)
      exacts [Or.inl rfl, Or.inr rfl]
    let eX : { p : Z × PUnit // f p.fst = Sum.inl p.snd } ≃ { x : Z // f x = Sum.inl PUnit.unit } :=
      ⟨fun p => ⟨p.1.1, by convert p.2⟩, fun x => ⟨⟨_, _⟩, x.2⟩, fun _ => by ext; rfl,
        fun _ => by ext; rfl⟩
    let eY : { p : Z × PUnit // f p.fst = Sum.inr p.snd } ≃ { x : Z // f x = Sum.inr PUnit.unit } :=
      ⟨fun p => ⟨p.1.1, p.2.trans (congr_arg Sum.inr <| Subsingleton.elim _ _)⟩,
        fun x => ⟨⟨_, _⟩, x.2⟩, fun _ => by ext; rfl, fun _ => by ext; rfl⟩
    fapply BinaryCofan.isColimitMk
    · exact fun s x => dite _ (fun h => s.inl <| eX.symm ⟨x, h⟩)
        fun h => s.inr <| eY.symm ⟨x, (this x).resolve_left h⟩
    · intro s
      ext ⟨⟨x, ⟨⟩⟩, _⟩
      dsimp
      split_ifs <;> rfl
    · intro s
      ext ⟨⟨x, ⟨⟩⟩, hx⟩
      dsimp
      split_ifs with h
      · cases h.symm.trans hx
      · rfl
    · intro s m e₁ e₂
      ext x
      split_ifs
      · rw [← e₁]
        rfl
      · rw [← e₂]
        rfl


/-- (Implementation) An auxiliary lemma for the proof that `TopCat` is finitary extensive. -/
noncomputable def finitaryExtensiveTopCatAux (Z : TopCat.{u})
    (f : Z ⟶ TopCat.of (PUnit.{u + 1} ⊕ PUnit.{u + 1})) :
    IsColimit (BinaryCofan.mk
      (TopCat.pullbackFst f (TopCat.binaryCofan (TopCat.of PUnit) (TopCat.of PUnit)).inl)
      (TopCat.pullbackFst f (TopCat.binaryCofan (TopCat.of PUnit) (TopCat.of PUnit)).inr)) := by
  have h₁ : Set.range (TopCat.pullbackFst f (TopCat.binaryCofan (.of PUnit) (.of PUnit)).inl) =
      f ⁻¹' Set.range Sum.inl := by
    apply le_antisymm
    · rintro _ ⟨x, rfl⟩; exact ⟨PUnit.unit, x.2.symm⟩
    · rintro x ⟨⟨⟩, hx⟩; refine ⟨⟨⟨x, PUnit.unit⟩, hx.symm⟩, rfl⟩
  have h₂ : Set.range (TopCat.pullbackFst f (TopCat.binaryCofan (.of PUnit) (.of PUnit)).inr) =
      f ⁻¹' Set.range Sum.inr := by
    apply le_antisymm
    · rintro _ ⟨x, rfl⟩; exact ⟨PUnit.unit, x.2.symm⟩
    · rintro x ⟨⟨⟩, hx⟩; refine ⟨⟨⟨x, PUnit.unit⟩, hx.symm⟩, rfl⟩
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} D
    X Y : C
    Z : TopCat
    f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
    h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
    h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (TopCa …
  -/
  refine ((TopCat.binaryCofan_isColimit_iff _).mpr ⟨?_, ?_, ?_⟩).some
    /-
      case refine_1
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
      h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.Limits.BinaryCofan.mk (TopCat.pull …
    -/
  · refine ⟨(Homeomorph.prodPUnit Z).isEmbedding.comp .subtypeVal, ?_⟩
    /-
      case refine_1
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
      h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      ⊢ IsOpen (Set.range ⇑(CategoryTheory.Limits.BinaryCofan.mk (TopCat.pullbackFst …
    -/
    convert f.2.1 _ isOpen_range_inl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
      h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.Limits.BinaryCofan.mk (TopCat.pull …
    -/
  · refine ⟨(Homeomorph.prodPUnit Z).isEmbedding.comp .subtypeVal, ?_⟩
    /-
      case refine_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
      h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      ⊢ IsOpen (Set.range ⇑(CategoryTheory.Limits.BinaryCofan.mk (TopCat.pullbackFst …
    -/
    convert f.2.1 _ isOpen_range_inr
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z (TopCat.of (Sum PUnit.{u + 1} PUnit.{u + 1}))
      h₁ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      h₂ : Eq (Set.range ⇑(TopCat.pullbackFst f ((TopCat.of PUnit.{u + 1}).binaryCof …
      ⊢ IsCompl (Set.range ⇑(CategoryTheory.Limits.BinaryCofan.mk (TopCat.pullbackFs …
    -/
  · convert Set.isCompl_range_inl_range_inr.preimage f
    /-
      🎉 no goals
    -/


instance finitaryExtensive_TopCat : FinitaryExtensive TopCat.{u} := by
  rw [finitaryExtensive_iff_of_isTerminal TopCat.{u} _ TopCat.isTerminalPUnit _
      (TopCat.binaryCofanIsColimit _ _)]
  apply BinaryCofan.isVanKampen_mk _ _ (fun X Y => TopCat.binaryCofanIsColimit X Y) _
      fun f g => TopCat.pullbackConeIsLimit f g
    /-
      case h₁
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      ⊢ ∀ {X' Y' : TopCat} (αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})) (αY : Quiv …
    -/
  · intro X' Y' αX αY f hαX hαY
    /-
      case h₁
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      X' Y' : TopCat
      αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
      αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
      f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
      hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
      hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
      ⊢ And (CategoryTheory.IsPullback (X'.binaryCofan Y').inl αX f ((TopCat.of PUni …
    -/
    constructor
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        ⊢ CategoryTheory.IsPullback (X'.binaryCofan Y').inl αX f ((TopCat.of PUnit.{u  …
      -/
    · refine ⟨⟨hαX.symm⟩, ⟨PullbackCone.isLimitAux' _ ?_⟩⟩
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        ⊢ (s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryC …
      -/
      intro s
      have : ∀ x, ∃! y, s.fst x = Sum.inl y := by
        intro x
        cases' h : s.fst x with val val
        · exact ⟨val, rfl, fun y h => Sum.inl_injective h.symm⟩
        · apply_fun f at h
          cases ((ConcreteCategory.congr_hom s.condition x).symm.trans h).trans
            (ConcreteCategory.congr_hom hαY val : _).symm
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        this : ∀ (x : ↑s.pt), ExistsUnique fun y => Eq (s.fst x) (Sum.inl y)
        ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
      -/
      delta ExistsUnique at this
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        this : ∀ (x : ↑s.pt), Exists fun x_1 => And ((fun y => Eq (s.fst x) (Sum.inl y …
        ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
      -/
      choose l hl hl' using this
      refine ⟨⟨l, ?_⟩, ContinuousMap.ext fun a => (hl a).symm, TopCat.isTerminalPUnit.hom_ext _ _,
        fun {l'} h₁ _ => ContinuousMap.ext fun x =>
          hl' x (l' x) (ConcreteCategory.congr_hom h₁ x).symm⟩
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑X'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inl y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑X'), (fun y => Eq (s.fst x) (Sum.inl y)) y → Eq y (l …
        ⊢ Continuous l
      -/
      apply (IsEmbedding.inl (X := X') (Y := Y')).isInducing.continuous_iff.mpr
      /-
        case h₁.left
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑X'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inl y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑X'), (fun y => Eq (s.fst x) (Sum.inl y)) y → Eq y (l …
        ⊢ Continuous (Function.comp Sum.inl l)
      -/
      convert s.fst.2 using 1
      /-
        case h.e'_5.h
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑X'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inl y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑X'), (fun y => Eq (s.fst x) (Sum.inl y)) y → Eq y (l …
        e_2✝ : Eq (Sum ↑X' ↑Y') ↑(((CategoryTheory.Functor.const (CategoryTheory.Discr …
        ⊢ Eq (Function.comp Sum.inl l) s.fst.toFun
      -/
      exact (funext hl).symm
      /-
        🎉 no goals
      -/
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        ⊢ CategoryTheory.IsPullback (X'.binaryCofan Y').inr αY f ((TopCat.of PUnit.{u  …
      -/
    · refine ⟨⟨hαY.symm⟩, ⟨PullbackCone.isLimitAux' _ ?_⟩⟩
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        ⊢ (s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryC …
      -/
      intro s
      have : ∀ x, ∃! y, s.fst x = Sum.inr y := by
        intro x
        cases' h : s.fst x with val val
        · apply_fun f at h
          cases ((ConcreteCategory.congr_hom s.condition x).symm.trans h).trans
            (ConcreteCategory.congr_hom hαX val : _).symm
        · exact ⟨val, rfl, fun y h => Sum.inr_injective h.symm⟩
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        this : ∀ (x : ↑s.pt), ExistsUnique fun y => Eq (s.fst x) (Sum.inr y)
        ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
      -/
      delta ExistsUnique at this
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        this : ∀ (x : ↑s.pt), Exists fun x_1 => And ((fun y => Eq (s.fst x) (Sum.inr y …
        ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
      -/
      choose l hl hl' using this
      refine ⟨⟨l, ?_⟩, ContinuousMap.ext fun a => (hl a).symm, TopCat.isTerminalPUnit.hom_ext _ _,
        fun {l'} h₁ _ =>
          ContinuousMap.ext fun x => hl' x (l' x) (ConcreteCategory.congr_hom h₁ x).symm⟩
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑Y'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inr y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑Y'), (fun y => Eq (s.fst x) (Sum.inr y)) y → Eq y (l …
        ⊢ Continuous l
      -/
      apply (IsEmbedding.inr (X := X') (Y := Y')).isInducing.continuous_iff.mpr
      /-
        case h₁.right
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑Y'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inr y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑Y'), (fun y => Eq (s.fst x) (Sum.inr y)) y → Eq y (l …
        ⊢ Continuous (Function.comp Sum.inr l)
      -/
      convert s.fst.2 using 1
      /-
        case h.e'_5.h
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u''
        inst✝ : CategoryTheory.Category.{v'', u''} D
        X Y : C
        X' Y' : TopCat
        αX : Quiver.Hom X' (TopCat.of PUnit.{u + 1})
        αY : Quiver.Hom Y' (TopCat.of PUnit.{u + 1})
        f : Quiver.Hom (X'.binaryCofan Y').pt ((TopCat.of PUnit.{u + 1}).binaryCofan ( …
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX ((TopCat.of PUnit.{u + 1}).bin …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY ((TopCat.of PUnit.{u + 1}).bin …
        s : CategoryTheory.Limits.PullbackCone f ((TopCat.of PUnit.{u + 1}).binaryCofa …
        l : ↑s.pt → ↑Y'
        hl : ∀ (x : ↑s.pt), (fun y => Eq (s.fst x) (Sum.inr y)) (l x)
        hl' : ∀ (x : ↑s.pt) (y : ↑Y'), (fun y => Eq (s.fst x) (Sum.inr y)) y → Eq y (l …
        e_2✝ : Eq (Sum ↑X' ↑Y') ↑(((CategoryTheory.Functor.const (CategoryTheory.Discr …
        ⊢ Eq (Function.comp Sum.inr l) s.fst.toFun
      -/
      exact (funext hl).symm
      /-
        🎉 no goals
      -/
    /-
      case h₂
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      ⊢ {Z : TopCat} → (f : Quiver.Hom Z ((TopCat.of PUnit.{u + 1}).binaryCofan (Top …
    -/
  · intro Z f
    /-
      case h₂
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝ : CategoryTheory.Category.{v'', u''} D
      X Y : C
      Z : TopCat
      f : Quiver.Hom Z ((TopCat.of PUnit.{u + 1}).binaryCofan (TopCat.of PUnit.{u +  …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (TopCa …
    -/
    exact finitaryExtensiveTopCatAux Z f
    /-
      🎉 no goals
    -/


theorem finitaryExtensive_of_reflective
    [HasFiniteCoproducts D] [HasPullbacksOfInclusions D] [FinitaryExtensive C]
    {Gl : C ⥤ D} {Gr : D ⥤ C} (adj : Gl ⊣ Gr) [Gr.Full] [Gr.Faithful]
    [∀ X Y (f : X ⟶ Gl.obj Y), HasPullback (Gr.map f) (adj.unit.app Y)]
    [∀ X Y (f : X ⟶ Gl.obj Y), PreservesLimit (cospan (Gr.map f) (adj.unit.app Y)) Gl]
    [PreservesPullbacksOfInclusions Gl] :
    FinitaryExtensive D := by
  /-
    C : Type u
    inst✝⁹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁸ : CategoryTheory.Category.{v'', u''} D
    inst✝⁷ : CategoryTheory.Limits.HasFiniteCoproducts D
    inst✝⁶ : CategoryTheory.HasPullbacksOfInclusions D
    inst✝⁵ : CategoryTheory.FinitaryExtensive C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    inst✝² : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝¹ : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝ : CategoryTheory.PreservesPullbacksOfInclusions Gl
    ⊢ CategoryTheory.FinitaryExtensive D
  -/
  have : PreservesColimitsOfSize Gl := adj.leftAdjoint_preservesColimits
  /-
    C : Type u
    inst✝⁹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁸ : CategoryTheory.Category.{v'', u''} D
    inst✝⁷ : CategoryTheory.Limits.HasFiniteCoproducts D
    inst✝⁶ : CategoryTheory.HasPullbacksOfInclusions D
    inst✝⁵ : CategoryTheory.FinitaryExtensive C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    inst✝² : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝¹ : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝ : CategoryTheory.PreservesPullbacksOfInclusions Gl
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.98708, ?u.98707, v, v …
    ⊢ CategoryTheory.FinitaryExtensive D
  -/
  constructor
  /-
    case van_kampen'
    C : Type u
    inst✝⁹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁸ : CategoryTheory.Category.{v'', u''} D
    inst✝⁷ : CategoryTheory.Limits.HasFiniteCoproducts D
    inst✝⁶ : CategoryTheory.HasPullbacksOfInclusions D
    inst✝⁵ : CategoryTheory.FinitaryExtensive C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    inst✝² : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝¹ : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
    inst✝ : CategoryTheory.PreservesPullbacksOfInclusions Gl
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.98708, ?u.98707, v, v …
    ⊢ ∀ {X Y : D} (c : CategoryTheory.Limits.BinaryCofan X Y), CategoryTheory.Limi …
  -/
  intros X Y c hc
  apply (IsVanKampenColimit.precompose_isIso_iff
    (isoWhiskerLeft _ (asIso adj.counit) ≪≫ Functor.rightUnitor _).hom).mp
  have : ∀ (Z : C) (i : Discrete WalkingPair) (f : Z ⟶ (colimit.cocone (pair X Y ⋙ Gr)).pt),
        PreservesLimit (cospan f ((colimit.cocone (pair X Y ⋙ Gr)).ι.app i)) Gl := by
    have : pair X Y ⋙ Gr = pair (Gr.obj X) (Gr.obj Y) := by
      apply Functor.hext
      · rintro ⟨⟨⟩⟩ <;> rfl
      · rintro ⟨⟨⟩⟩ ⟨j⟩ ⟨⟨rfl : _ = j⟩⟩ <;> simp
    rw [this]
    rintro Z ⟨_|_⟩ f <;> dsimp <;> infer_instance
  refine ((FinitaryExtensive.vanKampen _ (colimit.isColimit <| pair X Y ⋙ _)).map_reflective
    adj).of_iso (IsColimit.uniqueUpToIso ?_ ?_)
    /-
      case van_kampen'.refine_1
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      inst✝⁷ : CategoryTheory.Limits.HasFiniteCoproducts D
      inst✝⁶ : CategoryTheory.HasPullbacksOfInclusions D
      inst✝⁵ : CategoryTheory.FinitaryExtensive C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      inst✝² : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
      inst✝¹ : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
      inst✝ : CategoryTheory.PreservesPullbacksOfInclusions Gl
      this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.98708, ?u.98707, v,  …
      X Y : D
      c : CategoryTheory.Limits.BinaryCofan X Y
      hc : CategoryTheory.Limits.IsColimit c
      this : ∀ (Z : C) (i : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPai …
      ⊢ CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.colimit …
    -/
  · exact isColimitOfPreserves Gl (colimit.isColimit _)
    /-
      🎉 no goals
    -/
    /-
      case van_kampen'.refine_2
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      inst✝⁷ : CategoryTheory.Limits.HasFiniteCoproducts D
      inst✝⁶ : CategoryTheory.HasPullbacksOfInclusions D
      inst✝⁵ : CategoryTheory.FinitaryExtensive C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      inst✝² : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
      inst✝¹ : ∀ (X : D) (Y : C) (f : Quiver.Hom X (Gl.obj Y)), CategoryTheory.Limit …
      inst✝ : CategoryTheory.PreservesPullbacksOfInclusions Gl
      this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v'', u, u''} Gl
      X Y : D
      c : CategoryTheory.Limits.BinaryCofan X Y
      hc : CategoryTheory.Limits.IsColimit c
      this : ∀ (Z : C) (i : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPai …
      ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose ( …
    -/
  · exact (IsColimit.precomposeHomEquiv _ _).symm hc
    /-
      🎉 no goals
    -/


instance finitaryExtensive_functor [HasPullbacks C] [FinitaryExtensive C] :
    FinitaryExtensive (D ⥤ C) :=
  haveI : HasFiniteCoproducts (D ⥤ C) := ⟨fun _ => Limits.functorCategoryHasColimitsOfShape⟩
  ⟨fun c hc => isVanKampenColimit_of_evaluation _ c fun _ =>
    FinitaryExtensive.vanKampen _ <| isColimitOfPreserves _ hc⟩


instance {C} [Category C] {D} [Category D] (F : C ⥤ D)
    {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [IsIso f] : PreservesLimit (cospan f g) F :=
  have := hasPullback_of_left_iso f g
  preservesLimit_of_preserves_limit_cone (IsPullback.of_hasPullback f g).isLimit
    ((isLimitMapConePullbackConeEquiv _ pullback.condition).symm
                                    /-
                                      J : Type v'
                                      inst✝⁵ : CategoryTheory.Category.{u', v'} J
                                      C✝ : Type u
                                      inst✝⁴ : CategoryTheory.Category.{v, u} C✝
                                      D✝ : Type u''
                                      inst✝³ : CategoryTheory.Category.{v'', u''} D✝
                                      X✝ Y✝ : C✝
                                      C : Type u_1
                                      inst✝² : CategoryTheory.Category.{u_2, u_1} C
                                      D : Type u_3
                                      inst✝¹ : CategoryTheory.Category.{u_4, u_3} D
                                      F : CategoryTheory.Functor C D
                                      X Y Z : C
                                      f : Quiver.Hom X Z
                                      g : Quiver.Hom Y Z
                                      inst✝ : CategoryTheory.IsIso f
                                      this : CategoryTheory.Limits.HasPullback f g
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.pullbac …
                                    -/
      (IsPullback.of_vert_isIso ⟨by simp only [← F.map_comp, pullback.condition]⟩).isLimit)
                                    /-
                                      🎉 no goals
                                    -/


instance {C} [Category C] {D} [Category D] (F : C ⥤ D)
    {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [IsIso g] : PreservesLimit (cospan f g) F :=
  preservesPullback_symmetry _ _ _


theorem finitaryExtensive_of_preserves_and_reflects (F : C ⥤ D) [FinitaryExtensive D]
    [HasFiniteCoproducts C] [HasPullbacksOfInclusions C]
    [PreservesPullbacksOfInclusions F]
    [ReflectsLimitsOfShape WalkingCospan F] [PreservesColimitsOfShape (Discrete WalkingPair) F]
    [ReflectsColimitsOfShape (Discrete WalkingPair) F] : FinitaryExtensive C := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.FinitaryExtensive D
    inst✝⁵ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝⁴ : CategoryTheory.HasPullbacksOfInclusions C
    inst✝³ : CategoryTheory.PreservesPullbacksOfInclusions F
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape (CategoryTheory.Discrete …
    ⊢ CategoryTheory.FinitaryExtensive C
  -/
  constructor
  /-
    case van_kampen'
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.FinitaryExtensive D
    inst✝⁵ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝⁴ : CategoryTheory.HasPullbacksOfInclusions C
    inst✝³ : CategoryTheory.PreservesPullbacksOfInclusions F
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape (CategoryTheory.Discrete …
    ⊢ ∀ {X Y : C} (c : CategoryTheory.Limits.BinaryCofan X Y), CategoryTheory.Limi …
  -/
  intros X Y c hc
  /-
    case van_kampen'
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.FinitaryExtensive D
    inst✝⁵ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝⁴ : CategoryTheory.HasPullbacksOfInclusions C
    inst✝³ : CategoryTheory.PreservesPullbacksOfInclusions F
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape (CategoryTheory.Discrete …
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  refine IsVanKampenColimit.of_iso ?_ (hc.uniqueUpToIso (coprodIsCoprod X Y)).symm
  have (i : Discrete WalkingPair) (Z : C) (f : Z ⟶ X ⨿ Y) :
    PreservesLimit (cospan f ((BinaryCofan.mk coprod.inl coprod.inr).ι.app i)) F := by
    rcases i with ⟨_|_⟩ <;> dsimp <;> infer_instance
  refine (FinitaryExtensive.vanKampen _
    (isColimitOfPreserves F (coprodIsCoprod X Y))).of_mapCocone F


theorem finitaryExtensive_of_preserves_and_reflects_isomorphism (F : C ⥤ D) [FinitaryExtensive D]
    [HasFiniteCoproducts C] [HasPullbacks C] [PreservesLimitsOfShape WalkingCospan F]
    [PreservesColimitsOfShape (Discrete WalkingPair) F] [F.ReflectsIsomorphisms] :
    FinitaryExtensive C := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁶ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : CategoryTheory.FinitaryExtensive D
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasPullbacks C
    inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : F.ReflectsIsomorphisms
    ⊢ CategoryTheory.FinitaryExtensive C
  -/
  haveI : ReflectsLimitsOfShape WalkingCospan F := reflectsLimitsOfShape_of_reflectsIsomorphisms
  haveI : ReflectsColimitsOfShape (Discrete WalkingPair) F :=
    reflectsColimitsOfShape_of_reflectsIsomorphisms
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁶ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : CategoryTheory.FinitaryExtensive D
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasPullbacks C
    inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : F.ReflectsIsomorphisms
    this✝ : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Walk …
    this : CategoryTheory.Limits.ReflectsColimitsOfShape (CategoryTheory.Discrete  …
    ⊢ CategoryTheory.FinitaryExtensive C
  -/
  exact finitaryExtensive_of_preserves_and_reflects F
  /-
    🎉 no goals
  -/


theorem FinitaryPreExtensive.isUniversal_finiteCoproducts_Fin [FinitaryPreExtensive C] {n : ℕ}
    {F : Discrete (Fin n) ⥤ C} {c : Cocone F} (hc : IsColimit c) : IsUniversalColimit c := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsUniversalColimit c
  -/
  let f : Fin n → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor f :=
    Functor.hext (fun _ ↦ rfl) (by rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩; simp [f])
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    f : Fin n → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor f)
    ⊢ CategoryTheory.IsUniversalColimit c
  -/
  clear_value f
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    f : Fin n → C
    this : Eq F (CategoryTheory.Discrete.functor f)
    ⊢ CategoryTheory.IsUniversalColimit c
  -/
  subst this
  induction n with
  | zero => exact (isVanKampenColimit_of_isEmpty _ hc).isUniversal
  | succ n IH =>
    refine IsUniversalColimit.of_iso (@isUniversalColimit_extendCofan _ _ _ _ _ _
      (IH _ (coproductIsCoproduct _)) (FinitaryPreExtensive.universal' _ (coprodIsCoprod _ _)) ?_)
      ((extendCofanIsColimit f (coproductIsCoproduct _) (coprodIsCoprod _ _)).uniqueUpToIso hc)
    · dsimp
      infer_instance


theorem FinitaryPreExtensive.isUniversal_finiteCoproducts [FinitaryPreExtensive C] {ι : Type*}
    [Finite ι] {F : Discrete ι ⥤ C} {c : Cocone F} (hc : IsColimit c) : IsUniversalColimit c := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsUniversalColimit c
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin ι
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.IsUniversalColimit c
  -/
  apply (IsUniversalColimit.whiskerEquivalence_iff (Discrete.equivalence e).symm).mp
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.Cocone.whisker (Cat …
  -/
  apply FinitaryPreExtensive.isUniversal_finiteCoproducts_Fin
  /-
    case intro.intro.hc
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker (Categ …
  -/
  exact (IsColimit.whiskerEquivalenceEquiv (Discrete.equivalence e).symm) hc
  /-
    🎉 no goals
  -/


theorem FinitaryExtensive.isVanKampen_finiteCoproducts_Fin [FinitaryExtensive C] {n : ℕ}
    {F : Discrete (Fin n) ⥤ C} {c : Cocone F} (hc : IsColimit c) : IsVanKampenColimit c := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  let f : Fin n → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor f :=
    Functor.hext (fun _ ↦ rfl) (by rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩; simp [f])
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    f : Fin n → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor f)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  clear_value f
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    n : Nat
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    f : Fin n → C
    this : Eq F (CategoryTheory.Discrete.functor f)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  subst this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.FinitaryExtensive C
    n : Nat
    f : Fin n → C
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  induction' n with n IH
    /-
      case zero
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.FinitaryExtensive C
      f : Fin 0 → C
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.IsVanKampenColimit c
    -/
  · exact isVanKampenColimit_of_isEmpty _ hc
    /-
      🎉 no goals
    -/
  · apply IsVanKampenColimit.of_iso _
      ((extendCofanIsColimit f (coproductIsCoproduct _) (coprodIsCoprod _ _)).uniqueUpToIso hc)
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.FinitaryExtensive C
      n : Nat
      IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
      f : Fin (HAdd.hAdd n 1) → C
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.extendCofan (CategoryTheor …
    -/
    apply @isVanKampenColimit_extendCofan _ _ _ _ _ _ _ _ ?_
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.Cofan.mk (CategoryT …
      -/
    · apply IH
      /-
        case hc
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (CategoryThe …
      -/
      exact coproductIsCoproduct _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.BinaryCofan.mk Cate …
      -/
    · apply FinitaryExtensive.van_kampen'
      /-
        case a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk Catego …
      -/
      exact coprodIsCoprod _ _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ ∀ {Z : C} (i : Quiver.Hom Z (CategoryTheory.Limits.BinaryCofan.mk CategoryTh …
      -/
    · dsimp
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.FinitaryExtensive C
        n : Nat
        IH : ∀ (f : Fin n → C) {c : CategoryTheory.Limits.Cocone (CategoryTheory.Discr …
        f : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ ∀ {Z : C} (i : Quiver.Hom Z (CategoryTheory.Limits.coprod (f 0) (CategoryThe …
      -/
      infer_instance
      /-
        🎉 no goals
      -/


theorem FinitaryExtensive.isVanKampen_finiteCoproducts [FinitaryExtensive C] {ι : Type*}
    [Finite ι] {F : Discrete ι ⥤ C} {c : Cocone F} (hc : IsColimit c) : IsVanKampenColimit c := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin ι
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  apply (IsVanKampenColimit.whiskerEquivalence_iff (Discrete.equivalence e).symm).mp
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.Cocone.whisker (Cat …
  -/
  apply FinitaryExtensive.isVanKampen_finiteCoproducts_Fin
  /-
    case intro.intro.hc
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    ι : Type u_1
    inst✝ : Finite ι
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker (Categ …
  -/
  exact (IsColimit.whiskerEquivalenceEquiv (Discrete.equivalence e).symm) hc
  /-
    🎉 no goals
  -/


lemma FinitaryPreExtensive.hasPullbacks_of_is_coproduct [FinitaryPreExtensive C] {ι : Type*}
    [Finite ι] {F : Discrete ι ⥤ C} {c : Cocone F} (hc : IsColimit c) (i : Discrete ι) {X : C}
    (g : X ⟶ _) : HasPullback g (c.ι.app i) := by
  classical
  let f : ι → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor f :=
    Functor.hext (fun i ↦ rfl) (by rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩; simp [f])
  clear_value f
  subst this
  change Cofan f at c
  obtain ⟨i⟩ := i
  let e : ∐ f ≅ f i ⨿ (∐ fun j : ({i}ᶜ : Set ι) ↦ f j) :=
  { hom := Sigma.desc (fun j ↦ if h : j = i then eqToHom (congr_arg f h) ≫ coprod.inl else
      Sigma.ι (fun j : ({i}ᶜ : Set ι) ↦ f j) ⟨j, h⟩ ≫ coprod.inr)
    inv := coprod.desc (Sigma.ι f i) (Sigma.desc fun j ↦ Sigma.ι f j)
    hom_inv_id := by aesop_cat
    inv_hom_id := by
      ext j
      · simp
      · simp only [coprod.desc_comp, colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app,
          eqToHom_refl, Category.id_comp, dite_true, BinaryCofan.mk_pt, BinaryCofan.ι_app_right,
          BinaryCofan.mk_inr, colimit.ι_desc_assoc, Discrete.functor_obj, Category.comp_id]
        exact dif_neg j.prop }
  let e' : c.pt ≅ f i ⨿ (∐ fun j : ({i}ᶜ : Set ι) ↦ f j) :=
    hc.coconePointUniqueUpToIso (getColimitCocone _).2 ≪≫ e
  have : coprod.inl ≫ e'.inv = c.ι.app ⟨i⟩ := by
    simp only [e, e', Iso.trans_inv, coprod.desc_comp, colimit.ι_desc, BinaryCofan.mk_pt,
      BinaryCofan.ι_app_left, BinaryCofan.mk_inl]
    exact colimit.comp_coconePointUniqueUpToIso_inv _ _
  clear_value e'
  rw [← this]
  have : IsPullback (𝟙 _) (g ≫ e'.hom) g e'.inv := IsPullback.of_horiz_isIso ⟨by simp⟩
  exact ⟨⟨⟨_, ((IsPullback.of_hasPullback (g ≫ e'.hom) coprod.inl).paste_horiz this).isLimit⟩⟩⟩


lemma FinitaryExtensive.mono_ι [FinitaryExtensive C] {ι : Type*} [Finite ι] {F : Discrete ι ⥤ C}
    {c : Cocone F} (hc : IsColimit c) (i : Discrete ι) :
    Mono (c.ι.app i) :=
  mono_of_cofan_isVanKampen (isVanKampen_finiteCoproducts hc) _


instance [FinitaryExtensive C] {ι : Type*} [Finite ι] (X : ι → C) (i : ι) :
    Mono (Sigma.ι X i) :=
  FinitaryExtensive.mono_ι (coproductIsCoproduct _) ⟨i⟩


lemma FinitaryExtensive.isPullback_initial_to [FinitaryExtensive C]
    {ι : Type*} [Finite ι] {F : Discrete ι ⥤ C}
    {c : Cocone F} (hc : IsColimit c) (i j : Discrete ι) (e : i ≠ j) :
    IsPullback (initial.to _) (initial.to _) (c.ι.app i) (c.ι.app j) :=
  isPullback_initial_to_of_cofan_isVanKampen (isVanKampen_finiteCoproducts hc) i j e


lemma FinitaryExtensive.isPullback_initial_to_sigma_ι [FinitaryExtensive C] {ι : Type*} [Finite ι]
    (X : ι → C) (i j : ι) (e : i ≠ j) :
    IsPullback (initial.to _) (initial.to _) (Sigma.ι X i) (Sigma.ι X j) :=
  FinitaryExtensive.isPullback_initial_to (coproductIsCoproduct _) ⟨i⟩ ⟨j⟩
    (ne_of_apply_ne Discrete.as e)


instance FinitaryPreExtensive.hasPullbacks_of_inclusions [FinitaryPreExtensive C] {X Z : C}
    {α : Type*} (f : X ⟶ Z) {Y : (a : α) → C} (i : (a : α) → Y a ⟶ Z) [Finite α]
    [hi : IsIso (Sigma.desc i)] (a : α) : HasPullback f (i a) := by
  /-
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝² : CategoryTheory.Category.{v'', u''} D
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    X Z : C
    α : Type u_1
    f : Quiver.Hom X Z
    Y : α → C
    i : (a : α) → Quiver.Hom (Y a) Z
    inst✝ : Finite α
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc i)
    a : α
    ⊢ CategoryTheory.Limits.HasPullback f (i a)
  -/
  apply FinitaryPreExtensive.hasPullbacks_of_is_coproduct (c := Cofan.mk Z i)
  /-
    case hc
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝² : CategoryTheory.Category.{v'', u''} D
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    X Z : C
    α : Type u_1
    f : Quiver.Hom X Z
    Y : α → C
    i : (a : α) → Quiver.Hom (Y a) Z
    inst✝ : Finite α
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc i)
    a : α
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z i)
  -/
  exact @IsColimit.ofPointIso (t := Cofan.mk Z i) (P := _) (i := hi)
  /-
    🎉 no goals
  -/


lemma FinitaryPreExtensive.sigma_desc_iso [FinitaryPreExtensive C] {α : Type} [Finite α] {X : C}
    {Z : α → C} (π : (a : α) → Z a ⟶ X) {Y : C} (f : Y ⟶ X) (hπ : IsIso (Sigma.desc π)) :
    IsIso (Sigma.desc ((fun _ ↦ pullback.fst _ _) : (a : α) → pullback f (π a) ⟶ _)) := by
  suffices IsColimit (Cofan.mk _ ((fun _ ↦ pullback.fst _ _) : (a : α) → pullback f (π a) ⟶ _)) by
    change IsIso (this.coconePointUniqueUpToIso (getColimitCocone _).2).inv
    infer_instance
  let this : IsColimit (Cofan.mk X π) := by
    refine @IsColimit.ofPointIso (t := Cofan.mk X π) (P := coproductIsCoproduct Z) (i := ?_)
    convert hπ
    simp [coproductIsCoproduct]
  refine (FinitaryPreExtensive.isUniversal_finiteCoproducts this
    (Cofan.mk _ ((fun _ ↦ pullback.fst _ _) : (a : α) → pullback f (π a) ⟶ _))
    (Discrete.natTrans fun i ↦ pullback.snd _ _) f ?_
    (NatTrans.equifibered_of_discrete _) ?_).some
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      α : Type
      inst✝ : Finite α
      X : C
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      Y : C
      f : Quiver.Hom Y X
      hπ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X π) := …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fun …
    -/
  · ext
    /-
      case refine_1.w.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      α : Type
      inst✝ : Finite α
      X : C
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      Y : C
      f : Quiver.Hom Y X
      hπ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X π) := …
      x✝ : CategoryTheory.Discrete α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fu …
    -/
    simp [pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      α : Type
      inst✝ : Finite α
      X : C
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      Y : C
      f : Quiver.Hom Y X
      hπ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X π) := …
      ⊢ ∀ (j : CategoryTheory.Discrete α), CategoryTheory.IsPullback ((CategoryTheor …
    -/
  · exact fun j ↦ IsPullback.of_hasPullback f (π j.as)
    /-
      🎉 no goals
    -/


