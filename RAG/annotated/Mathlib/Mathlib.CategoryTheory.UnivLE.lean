theorem UnivLE.ofEssSurj (w : (uliftFunctor.{u, v} : Type v ⥤ Type max u v).EssSurj) :
    UnivLE.{max u v, v} :=
  fun α ↦ by
    /-
      w : CategoryTheory.uliftFunctor.{u, v}.EssSurj
      α : Type (max u v)
      ⊢ Small.{v, max u v} α
    -/
    obtain ⟨a', ⟨m⟩⟩ := w.mem_essImage α
    /-
      case intro.intro
      w : CategoryTheory.uliftFunctor.{u, v}.EssSurj
      α : Type (max u v)
      a' : Type v
      m : CategoryTheory.Iso (CategoryTheory.uliftFunctor.{u, v}.obj a') α
      ⊢ Small.{v, max u v} α
    -/
    exact ⟨a', ⟨(Iso.toEquiv m).symm.trans Equiv.ulift⟩⟩
    /-
      🎉 no goals
    -/


instance EssSurj.ofUnivLE [UnivLE.{max u v, v}] :
    (uliftFunctor.{u, v} : Type v ⥤ Type max u v).EssSurj where
  mem_essImage α :=
    ⟨Shrink α, ⟨Equiv.toIso (Equiv.ulift.trans (equivShrink α).symm)⟩⟩


theorem UnivLE_iff_essSurj :
    UnivLE.{max u v, v} ↔ (uliftFunctor.{u, v} : Type v ⥤ Type max u v).EssSurj :=
  ⟨fun _ => inferInstance, fun w => UnivLE.ofEssSurj w⟩


instance [UnivLE.{max u v, v}] : uliftFunctor.{u, v}.IsEquivalence where


def UnivLE.witness [UnivLE.{max u v, v}] : Type u ⥤ Type v :=
  uliftFunctor.{v, u} ⋙ (uliftFunctor.{u, v}).inv


instance [UnivLE.{max u v, v}] : UnivLE.witness.{u, v}.Faithful :=
  inferInstanceAs <| Functor.Faithful (_ ⋙ _)


instance [UnivLE.{max u v, v}] : UnivLE.witness.{u, v}.Full :=
  inferInstanceAs <| Functor.Full (_ ⋙ _)

