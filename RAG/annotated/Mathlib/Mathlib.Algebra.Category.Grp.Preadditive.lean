instance (P Q : AddCommGrp) : AddCommGroup (P ⟶ Q) :=
  (inferInstance : AddCommGroup (AddMonoidHom P Q))

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/10688): this lemma was not necessary in mathlib

@[simp]
lemma hom_add_apply {P Q : AddCommGrp} (f g : P ⟶ Q) (x : P) : (f + g) x = f x + g x := rfl


instance : Preadditive AddCommGrp where


