lemma Sheaf.isPullback_square_op_map_yoneda_presheafToSheaf_yoneda_iff
    [HasWeakSheafify J (Type v)]
    (F : Sheaf J (Type v)) (sq : Square C) :
    (sq.op.map ((yoneda ⋙ presheafToSheaf J _).op ⋙ yoneda.obj F)).IsPullback ↔
      (sq.op.map F.val).IsPullback := by
  refine Square.IsPullback.iff_of_equiv _ _
    (((sheafificationAdjunction J (Type v)).homEquiv _ _).trans yonedaEquiv)
    (((sheafificationAdjunction J (Type v)).homEquiv _ _).trans yonedaEquiv)
    (((sheafificationAdjunction J (Type v)).homEquiv _ _).trans yonedaEquiv)
    (((sheafificationAdjunction J (Type v)).homEquiv _ _).trans yonedaEquiv) ?_ ?_ ?_ ?_
  all_goals
    ext x
    dsimp
    rw [yonedaEquiv_naturality]
    erw [Adjunction.homEquiv_naturality_left]
    rfl


/-- A Mayer-Vietoris square in a category `C` equipped with a Grothendieck
topology consists of a commutative square `f₁₂ ≫ f₂₄ = f₁₃ ≫ f₃₄` in `C`
such that `f₁₃` is a monomorphism and that the square becomes a
pushout square in the category of sheaves of sets. -/
structure MayerVietorisSquare [HasWeakSheafify J (Type v)] extends Square C where
  mono_f₁₃ : Mono toSquare.f₁₃ := by infer_instance
  /-- the square becomes a pushout square in the category of sheaves of types -/
  isPushout : (toSquare.map (yoneda ⋙ presheafToSheaf J _)).IsPushout


/-- Constructor for Mayer-Vietoris squares taking as an input
a square `sq` such that `sq.f₁₃` is a mono and that for every
sheaf of types `F`, the square `sq.op.map F.val` is a pullback square. -/
@[simps toSquare]
noncomputable def mk' (sq : Square C) [Mono sq.f₁₃]
    (H : ∀ (F : Sheaf J (Type v)), (sq.op.map F.val).IsPullback) :
    J.MayerVietorisSquare where
  toSquare := sq
  isPushout := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
      sq : CategoryTheory.Square C
      inst✝ : CategoryTheory.Mono sq.f₁₃
      H : ∀ (F : CategoryTheory.Sheaf J (Type v)), (sq.op.map F.val).IsPullback
      ⊢ (sq.map (CategoryTheory.yoneda.comp (CategoryTheory.presheafToSheaf J (Type  …
    -/
    rw [Square.isPushout_iff_op_map_yoneda_isPullback]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
      sq : CategoryTheory.Square C
      inst✝ : CategoryTheory.Mono sq.f₁₃
      H : ∀ (F : CategoryTheory.Sheaf J (Type v)), (sq.op.map F.val).IsPullback
      ⊢ ∀ (X : CategoryTheory.Sheaf J (Type v)), ((sq.map (CategoryTheory.yoneda.com …
    -/
    intro F
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
      sq : CategoryTheory.Square C
      inst✝ : CategoryTheory.Mono sq.f₁₃
      H : ∀ (F : CategoryTheory.Sheaf J (Type v)), (sq.op.map F.val).IsPullback
      F : CategoryTheory.Sheaf J (Type v)
      ⊢ ((sq.map (CategoryTheory.yoneda.comp (CategoryTheory.presheafToSheaf J (Type …
    -/
    exact (F.isPullback_square_op_map_yoneda_presheafToSheaf_yoneda_iff sq).2 (H F)
    /-
      🎉 no goals
    -/


/-- Constructor for Mayer-Vietoris squares taking as an input
a pullback square `sq` such that `sq.f₂₄` and `sq.f₃₄` are two monomorphisms
which form a covering of `S.X₄`. -/
@[simps! toSquare]
noncomputable def mk_of_isPullback (sq : Square C) [Mono sq.f₂₄] [Mono sq.f₃₄]
    (h₁ : sq.IsPullback) (h₂ : Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄ ∈ J sq.X₄) :
    J.MayerVietorisSquare :=
  have : Mono sq.f₁₃ := h₁.mono_f₁₃
  mk' sq (fun F ↦ by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
      sq : CategoryTheory.Square C
      inst✝¹ : CategoryTheory.Mono sq.f₂₄
      inst✝ : CategoryTheory.Mono sq.f₃₄
      h₁ : sq.IsPullback
      h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
      this : CategoryTheory.Mono sq.f₁₃
      F : CategoryTheory.Sheaf J (Type v)
      ⊢ (sq.op.map F.val).IsPullback
    -/
    apply Square.IsPullback.mk
    refine PullbackCone.IsLimit.mk _
      (fun s ↦ F.2.amalgamateOfArrows _ h₂
        (fun j ↦ WalkingPair.casesOn j s.fst s.snd)
        (fun W ↦ by
          rintro (_|_) (_|_) a b fac
          · obtain rfl : a = b := by simpa only [← cancel_mono sq.f₂₄] using fac
            rfl
          · obtain ⟨φ, rfl, rfl⟩ := PullbackCone.IsLimit.lift' h₁.isLimit _ _ fac
            simpa using s.condition =≫ F.val.map φ.op
          · obtain ⟨φ, rfl, rfl⟩ := PullbackCone.IsLimit.lift' h₁.isLimit _ _ fac.symm
            simpa using s.condition.symm =≫ F.val.map φ.op
          · obtain rfl : a = b := by simpa only [← cancel_mono sq.f₃₄] using fac
            rfl)) (fun _ ↦ ?_) (fun _ ↦ ?_) (fun s m hm₁ hm₂ ↦ ?_)
      /-
        case h.refine_1
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
        sq : CategoryTheory.Square C
        inst✝¹ : CategoryTheory.Mono sq.f₂₄
        inst✝ : CategoryTheory.Mono sq.f₃₄
        h₁ : sq.IsPullback
        h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
        this : CategoryTheory.Mono sq.f₁₃
        F : CategoryTheory.Sheaf J (Type v)
        x✝ : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => ⋯.amalgamateOfArrows (fun  …
      -/
    · exact F.2.amalgamateOfArrows_map _ _ _ _ WalkingPair.left
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
        sq : CategoryTheory.Square C
        inst✝¹ : CategoryTheory.Mono sq.f₂₄
        inst✝ : CategoryTheory.Mono sq.f₃₄
        h₁ : sq.IsPullback
        h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
        this : CategoryTheory.Mono sq.f₁₃
        F : CategoryTheory.Sheaf J (Type v)
        x✝ : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => ⋯.amalgamateOfArrows (fun  …
      -/
    · exact F.2.amalgamateOfArrows_map _ _ _ _ WalkingPair.right
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
        sq : CategoryTheory.Square C
        inst✝¹ : CategoryTheory.Mono sq.f₂₄
        inst✝ : CategoryTheory.Mono sq.f₃₄
        h₁ : sq.IsPullback
        h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
        this : CategoryTheory.Mono sq.f₁₃
        F : CategoryTheory.Sheaf J (Type v)
        s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
        m : Quiver.Hom s.pt (sq.op.map F.val).X₁
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
        ⊢ Eq m ((fun s => ⋯.amalgamateOfArrows (fun k => CategoryTheory.Limits.Walking …
      -/
    · apply F.2.hom_ext_ofArrows _ h₂
      /-
        case h.refine_3
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
        sq : CategoryTheory.Square C
        inst✝¹ : CategoryTheory.Mono sq.f₂₄
        inst✝ : CategoryTheory.Mono sq.f₃₄
        h₁ : sq.IsPullback
        h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
        this : CategoryTheory.Mono sq.f₁₃
        F : CategoryTheory.Sheaf J (Type v)
        s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
        m : Quiver.Hom s.pt (sq.op.map F.val).X₁
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
        ⊢ ∀ (i : CategoryTheory.Limits.WalkingPair), Eq (CategoryTheory.CategoryStruct …
      -/
      rintro (_|_)
        /-
          case h.refine_3.left
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
          sq : CategoryTheory.Square C
          inst✝¹ : CategoryTheory.Mono sq.f₂₄
          inst✝ : CategoryTheory.Mono sq.f₃₄
          h₁ : sq.IsPullback
          h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
          this : CategoryTheory.Mono sq.f₁₃
          F : CategoryTheory.Sheaf J (Type v)
          s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
          m : Quiver.Hom s.pt (sq.op.map F.val).X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Limits.W …
        -/
      · rw [F.2.amalgamateOfArrows_map _ _ _ _ WalkingPair.left]
        /-
          case h.refine_3.left
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
          sq : CategoryTheory.Square C
          inst✝¹ : CategoryTheory.Mono sq.f₂₄
          inst✝ : CategoryTheory.Mono sq.f₃₄
          h₁ : sq.IsPullback
          h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
          this : CategoryTheory.Mono sq.f₁₃
          F : CategoryTheory.Sheaf J (Type v)
          s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
          m : Quiver.Hom s.pt (sq.op.map F.val).X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Limits.W …
        -/
        exact hm₁
        /-
          🎉 no goals
        -/
        /-
          case h.refine_3.right
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
          sq : CategoryTheory.Square C
          inst✝¹ : CategoryTheory.Mono sq.f₂₄
          inst✝ : CategoryTheory.Mono sq.f₃₄
          h₁ : sq.IsPullback
          h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
          this : CategoryTheory.Mono sq.f₁₃
          F : CategoryTheory.Sheaf J (Type v)
          s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
          m : Quiver.Hom s.pt (sq.op.map F.val).X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Limits.W …
        -/
      · rw [F.2.amalgamateOfArrows_map _ _ _ _ WalkingPair.right]
        /-
          case h.refine_3.right
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          inst✝² : CategoryTheory.HasWeakSheafify J (Type v)
          sq : CategoryTheory.Square C
          inst✝¹ : CategoryTheory.Mono sq.f₂₄
          inst✝ : CategoryTheory.Mono sq.f₃₄
          h₁ : sq.IsPullback
          h₂ : Membership.mem (J sq.X₄) (CategoryTheory.Sieve.ofTwoArrows sq.f₂₄ sq.f₃₄)
          this : CategoryTheory.Mono sq.f₁₃
          F : CategoryTheory.Sheaf J (Type v)
          s : CategoryTheory.Limits.PullbackCone (sq.op.map F.val).f₂₄ (sq.op.map F.val) …
          m : Quiver.Hom s.pt (sq.op.map F.val).X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₂) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (sq.op.map F.val).f₁₃) s.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Limits.W …
        -/
        exact hm₂)
        /-
          🎉 no goals
        -/


lemma isPushoutAddCommGrpFreeSheaf [HasWeakSheafify J AddCommGrp.{v}] :
    (S.map (yoneda ⋙ (whiskeringRight _ _ _).obj AddCommGrp.free ⋙
      presheafToSheaf J _)).IsPushout :=
  (S.isPushout.map (Sheaf.composeAndSheafify J AddCommGrp.free)).of_iso
    ((Square.mapFunctor.mapIso
      (presheafToSheafCompComposeAndSheafifyIso J AddCommGrp.free)).app
        (S.map yoneda))


/-- The condition that a Mayer-Vietoris square becomes a pullback square
when we evaluate a presheaf on it. --/
def SheafCondition {A : Type u'} [Category.{v'} A] (P : Cᵒᵖ ⥤ A) : Prop :=
  (S.toSquare.op.map P).IsPullback


lemma sheafCondition_iff_comp_coyoneda {A : Type u'} [Category.{v'} A] (P : Cᵒᵖ ⥤ A) :
    S.SheafCondition P ↔ ∀ (X : Aᵒᵖ), S.SheafCondition (P ⋙ coyoneda.obj X) :=
  Square.isPullback_iff_map_coyoneda_isPullback (S.op.map P)


/-- Given a Mayer-Vietoris square `S` and a presheaf of types, this is the
map from `P.obj (op S.X₄)` to the explicit fibre product of
`P.map S.f₁₂.op` and `P.map S.f₁₃.op`. -/
abbrev toPullbackObj (P : Cᵒᵖ ⥤ Type v') :
    P.obj (op S.X₄) → Types.PullbackObj (P.map S.f₁₂.op) (P.map S.f₁₃.op) :=
  (S.toSquare.op.map P).pullbackCone.toPullbackObj


lemma sheafCondition_iff_bijective_toPullbackObj (P : Cᵒᵖ ⥤ Type v') :
    S.SheafCondition P ↔ Function.Bijective (S.toPullbackObj P) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : CategoryTheory.HasWeakSheafify J (Type v)
    S : J.MayerVietorisSquare
    P : CategoryTheory.Functor (Opposite C) (Type v')
    ⊢ Iff (S.SheafCondition P) (Function.Bijective (S.toPullbackObj P))
  -/
  have := (S.toSquare.op.map P).pullbackCone.isLimitEquivBijective
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : CategoryTheory.HasWeakSheafify J (Type v)
    S : J.MayerVietorisSquare
    P : CategoryTheory.Functor (Opposite C) (Type v')
    this : Equiv (CategoryTheory.Limits.IsLimit (S.op.map P).pullbackCone) (Functi …
    ⊢ Iff (S.SheafCondition P) (Function.Bijective (S.toPullbackObj P))
  -/
  exact ⟨fun h ↦ this h.isLimit, fun h ↦ Square.IsPullback.mk _ (this.symm h)⟩
  /-
    🎉 no goals
  -/


lemma bijective_toPullbackObj : Function.Bijective (S.toPullbackObj P) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : CategoryTheory.HasWeakSheafify J (Type v)
    S : J.MayerVietorisSquare
    P : CategoryTheory.Functor (Opposite C) (Type v')
    h : S.SheafCondition P
    ⊢ Function.Bijective (S.toPullbackObj P)
  -/
  rwa [← sheafCondition_iff_bijective_toPullbackObj]
  /-
    🎉 no goals
  -/


lemma ext {x y : P.obj (op S.X₄)}
    (h₁ : P.map S.f₂₄.op x = P.map S.f₂₄.op y)
    (h₂ : P.map S.f₃₄.op x = P.map S.f₃₄.op y) : x = y :=
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            J : CategoryTheory.GrothendieckTopology C
                                            inst✝ : CategoryTheory.HasWeakSheafify J (Type v)
                                            S : J.MayerVietorisSquare
                                            P : CategoryTheory.Functor (Opposite C) (Type v')
                                            h : S.SheafCondition P
                                            x y : P.obj { unop := S.X₄ }
                                            h₁ : Eq (P.map S.f₂₄.op x) (P.map S.f₂₄.op y)
                                            h₂ : Eq (P.map S.f₃₄.op x) (P.map S.f₃₄.op y)
                                            ⊢ Eq (S.toPullbackObj P x) (S.toPullbackObj P y)
                                          -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  h.bijective_toPullbackObj.injective (by ext <;> assumption)
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- If `S` is a Mayer-Vietoris square, and `P` is a presheaf
which satisfies the sheaf condition with respect to `S`, then
elements of `P` over `S.X₂` and `S.X₃` can be glued if the
coincide over `S.X₁`. -/
noncomputable def glue : P.obj (op S.X₄) :=
  (PullbackCone.IsLimit.equivPullbackObj h.isLimit).symm ⟨⟨u, v⟩, huv⟩


@[simp]
lemma map_f₂₄_op_glue : P.map S.f₂₄.op (h.glue u v huv) = u :=
  PullbackCone.IsLimit.equivPullbackObj_symm_apply_fst h.isLimit _


@[simp]
lemma map_f₃₄_op_glue : P.map S.f₃₄.op (h.glue u v huv) = v :=
  PullbackCone.IsLimit.equivPullbackObj_symm_apply_snd h.isLimit _


lemma sheafCondition_of_sheaf {A : Type u'} [Category.{v} A]
    (F : Sheaf J A) : S.SheafCondition F.val := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
    S : J.MayerVietorisSquare
    A : Type u'
    inst✝ : CategoryTheory.Category.{v, u'} A
    F : CategoryTheory.Sheaf J A
    ⊢ S.SheafCondition F.val
  -/
  rw [sheafCondition_iff_comp_coyoneda]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
    S : J.MayerVietorisSquare
    A : Type u'
    inst✝ : CategoryTheory.Category.{v, u'} A
    F : CategoryTheory.Sheaf J A
    ⊢ ∀ (X : Opposite A), S.SheafCondition (F.val.comp (CategoryTheory.coyoneda.ob …
  -/
  intro X
  exact (Sheaf.isPullback_square_op_map_yoneda_presheafToSheaf_yoneda_iff _ S.toSquare).1
    (S.isPushout.op.map
      (yoneda.obj ⟨_, (isSheaf_iff_isSheaf_of_type _ _).2 (F.cond X.unop)⟩))


/-- The short complex of abelian sheaves
`ℤ[S.X₁] ⟶ ℤ[S.X₂] ⊞ ℤ[S.X₃] ⟶ ℤ[S.X₄]`
where the left map is a difference and the right map a sum. -/
@[simps]
noncomputable def shortComplex :
    ShortComplex (Sheaf J AddCommGrp.{v}) where
  X₁ := (presheafToSheaf J _).obj (yoneda.obj S.X₁ ⋙ AddCommGrp.free)
  X₂ := (presheafToSheaf J _).obj (yoneda.obj S.X₂ ⋙ AddCommGrp.free) ⊞
    (presheafToSheaf J _).obj (yoneda.obj S.X₃ ⋙ AddCommGrp.free)
  X₃ := (presheafToSheaf J _).obj (yoneda.obj S.X₄ ⋙ AddCommGrp.free)
  f :=
    biprod.lift
      ((presheafToSheaf J _).map (whiskerRight (yoneda.map S.f₁₂) _))
      (-(presheafToSheaf J _).map (whiskerRight (yoneda.map S.f₁₃) _))
  g :=
    biprod.desc
      ((presheafToSheaf J _).map (whiskerRight (yoneda.map S.f₂₄) _))
      ((presheafToSheaf J _).map (whiskerRight (yoneda.map S.f₃₄) _))
  zero := (S.map (yoneda ⋙ (whiskeringRight _ _ _).obj AddCommGrp.free ⋙
      presheafToSheaf J _)).cokernelCofork.condition


instance : Mono S.shortComplex.f := by
  have : Mono (S.shortComplex.f ≫ biprod.snd) := by
    dsimp
    simp only [biprod.lift_snd]
    infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝¹ : CategoryTheory.HasWeakSheafify J (Type v)
    inst✝ : CategoryTheory.HasSheafify J AddCommGrp
    S : J.MayerVietorisSquare
    this : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp S.shortComplex. …
    ⊢ CategoryTheory.Mono S.shortComplex.f
  -/
  exact mono_of_mono _ biprod.snd
  /-
    🎉 no goals
  -/


instance : Epi S.shortComplex.g :=
  (S.shortComplex.exact_and_epi_g_iff_g_is_cokernel.2
    ⟨S.isPushoutAddCommGrpFreeSheaf.isColimitCokernelCofork⟩).2


lemma shortComplex_exact : S.shortComplex.Exact :=
  ShortComplex.exact_of_g_is_cokernel _
    S.isPushoutAddCommGrpFreeSheaf.isColimitCokernelCofork


lemma shortComplex_shortExact : S.shortComplex.ShortExact where
  exact := S.shortComplex_exact


