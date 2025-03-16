/-- A category `C` `IsSiftedOrEmpty` if the diagonal functor `C ⥤ C × C` is final. -/
abbrev IsSiftedOrEmpty : Prop := Final (diag C)


/-- A category `C` `IsSfited` if
1. the diagonal functor `C ⥤ C × C` is final.
2. there exists some object. -/
class IsSifted extends IsSiftedOrEmpty C : Prop where
  [nonempty : Nonempty C]


/-- Being sifted is preserved by equivalences of categories -/
lemma isSifted_of_equiv [IsSifted C] {D : Type u₁} [Category.{v₁} D] (e : D ≌ C) : IsSifted D :=
  letI : Final (diag D) := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.IsSifted C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      e : CategoryTheory.Equivalence D C
      ⊢ (CategoryTheory.Functor.diag D).Final
    -/
    letI : D × D ≌ C × C:= Equivalence.prod e e
    have sq : (e.inverse ⋙ diag D ⋙ this.functor ≅ diag C) :=
        NatIso.ofComponents (fun c ↦ by dsimp [this]
                                        exact Iso.prod (e.counitIso.app c) (e.counitIso.app c))
    apply_rules [final_iff_comp_equivalence _ this.functor|>.mpr,
      final_iff_final_comp e.inverse _ |>.mpr, final_of_natIso sq.symm]
  letI : _root_.Nonempty D := ⟨e.inverse.obj (_root_.Nonempty.some IsSifted.nonempty)⟩
  ⟨⟩


/-- In particular a category is sifted iff and only if it is so when viewed as a small category -/
lemma isSifted_iff_asSmallIsSifted : IsSifted C ↔ IsSifted (AsSmall.{w} C) where
  mp _ := isSifted_of_equiv AsSmall.equiv.symm
  mpr _ := isSifted_of_equiv AsSmall.equiv


/-- A sifted category is connected. -/
instance [IsSifted C] : IsConnected C :=
  isConnected_of_zigzag
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.IsSifted C
          ⊢ ∀ (j₁ j₂ : C), Exists fun l => And (List.Chain CategoryTheory.Zag j₁ l) (Eq  …
        -/
    (by intro c₁ c₂
        have X : StructuredArrow (c₁, c₂) (diag C) :=
          letI S : Final (diag C) := by infer_instance
          Nonempty.some (S.out (c₁, c₂)).is_nonempty
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.IsSifted C
          c₁ c₂ : C
          X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
          ⊢ Exists fun l => And (List.Chain CategoryTheory.Zag c₁ l) (Eq ((List.cons c₁  …
        -/
        use [X.right, c₂]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.IsSifted C
          c₁ c₂ : C
          X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
          ⊢ And (List.Chain CategoryTheory.Zag c₁ (List.cons X.right (List.cons c₂ List. …
        -/
        constructor
          /-
            case h.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.IsSifted C
            c₁ c₂ : C
            X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
            ⊢ List.Chain CategoryTheory.Zag c₁ (List.cons X.right (List.cons c₂ List.nil))
          -/
        · constructor
            /-
              case h.left.a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.IsSifted C
              c₁ c₂ : C
              X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
              ⊢ CategoryTheory.Zag c₁ X.right
            -/
          · exact Zag.of_hom X.hom.fst
            /-
              🎉 no goals
            -/
            /-
              case h.left.a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.IsSifted C
              c₁ c₂ : C
              X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
              ⊢ List.Chain CategoryTheory.Zag X.right (List.cons c₂ List.nil)
            -/
          · simp
            /-
              case h.left.a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.IsSifted C
              c₁ c₂ : C
              X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
              ⊢ CategoryTheory.Zag X.right c₂
            -/
            exact Zag.of_inv X.hom.snd
            /-
              🎉 no goals
            -/
          /-
            case h.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.IsSifted C
            c₁ c₂ : C
            X : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (CategoryTheory.Fu …
            ⊢ Eq ((List.cons c₁ (List.cons X.right (List.cons c₂ List.nil))).getLast ⋯) c₂
          -/
        · rfl)
          /-
            🎉 no goals
          -/


/-- A category with binary coproducts is sifted or empty. -/
instance [HasBinaryCoproducts C] : IsSiftedOrEmpty C := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      ⊢ CategoryTheory.IsSiftedOrEmpty C
    -/
    constructor
    /-
      case out
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      ⊢ ∀ (d : Prod C C), CategoryTheory.IsConnected (CategoryTheory.StructuredArrow …
    -/
    rintro ⟨c₁, c₂⟩
    haveI : _root_.Nonempty <| StructuredArrow (c₁,c₂) (diag C) :=
      ⟨.mk ((coprod.inl : c₁ ⟶ c₁ ⨿ c₂), (coprod.inr : c₂ ⟶ c₁ ⨿ c₂))⟩
    /-
      case out.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      c₁ c₂ : C
      this : Nonempty (CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (Cate …
      ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow { fst := c₁, snd  …
    -/
    apply isConnected_of_zigzag
    /-
      case out.mk.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      c₁ c₂ : C
      this : Nonempty (CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (Cate …
      ⊢ ∀ (j₁ j₂ : CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (Category …
    -/
    rintro ⟨_, c, f⟩ ⟨_, c', g⟩
    /-
      case out.mk.h.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      c₁ c₂ : C
      this : Nonempty (CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (Cate …
      left✝¹ : CategoryTheory.Discrete PUnit.{1}
      c : C
      f : Quiver.Hom ((CategoryTheory.Functor.fromPUnit { fst := c₁, snd := c₂ }).ob …
      left✝ : CategoryTheory.Discrete PUnit.{1}
      c' : C
      g : Quiver.Hom ((CategoryTheory.Functor.fromPUnit { fst := c₁, snd := c₂ }).ob …
      ⊢ Exists fun l => And (List.Chain CategoryTheory.Zag { left := left✝¹, right : …
    -/
    dsimp only [const_obj_obj, diag_obj, prod_Hom] at f g
    /-
      case out.mk.h.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      c₁ c₂ : C
      this : Nonempty (CategoryTheory.StructuredArrow { fst := c₁, snd := c₂ } (Cate …
      left✝¹ : CategoryTheory.Discrete PUnit.{1}
      c : C
      f : Prod (Quiver.Hom c₁ c) (Quiver.Hom c₂ c)
      left✝ : CategoryTheory.Discrete PUnit.{1}
      c' : C
      g : Prod (Quiver.Hom c₁ c') (Quiver.Hom c₂ c')
      ⊢ Exists fun l => And (List.Chain CategoryTheory.Zag { left := left✝¹, right : …
    -/
    use [.mk ((coprod.inl : c₁ ⟶ c₁ ⨿ c₂), (coprod.inr : c₂ ⟶ c₁ ⨿ c₂)), .mk (g.fst, g.snd)]
    simp only [colimit.cocone_x, diag_obj, Prod.mk.eta, List.chain_cons, List.Chain.nil, and_true,
      ne_eq, reduceCtorEq, not_false_eq_true, List.getLast_cons, List.cons_ne_self,
      List.getLast_singleton]
    exact ⟨⟨Zag.of_inv <| StructuredArrow.homMk <| coprod.desc f.fst f.snd,
      Zag.of_hom <| StructuredArrow.homMk <| coprod.desc g.fst g.snd⟩, rfl⟩


/-- A nonempty category with binary coproducts is sifted. -/
instance isSifted_of_hasBinaryCoproducts_and_nonempty [_root_.Nonempty C] [HasBinaryCoproducts C] :
    IsSifted C where


