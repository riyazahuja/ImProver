/-- A class for unbundled homs used to define a category. `hom` must
take two types `α`, `β` and instances of the corresponding structures,
and return a predicate on `α → β`. -/
class UnbundledHom {c : Type u → Type u} (hom : ∀ ⦃α β⦄, c α → c β → (α → β) → Prop) : Prop where
  hom_id : ∀ {α} (ia : c α), hom ia ia id
  hom_comp : ∀ {α β γ} {Iα : c α} {Iβ : c β} {Iγ : c γ} {g : β → γ} {f : α → β} (_ : hom Iβ Iγ g)
      (_ : hom Iα Iβ f), hom Iα Iγ (g ∘ f)


instance bundledHom : BundledHom fun α β (Iα : c α) (Iβ : c β) => Subtype (hom Iα Iβ) where
  toFun _ _ := Subtype.val
  id Iα := ⟨id, hom_id Iα⟩
  id_toFun _ := rfl
  comp _ _ _ g f := ⟨g.1 ∘ f.1, hom_comp g.2 f.2⟩
  comp_toFun _ _ _ _ _ := rfl
  hom_ext _ _ _ _ := Subtype.eq


/-- A custom constructor for forgetful functor
between concrete categories defined using `UnbundledHom`. -/
def mkHasForget₂ : HasForget₂ (Bundled c) (Bundled c') :=
  BundledHom.mkHasForget₂ obj (fun f => ⟨f.val, map f.property⟩) fun _ => rfl


