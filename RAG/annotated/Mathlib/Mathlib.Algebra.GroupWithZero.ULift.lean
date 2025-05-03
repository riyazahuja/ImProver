instance mulZeroOneClass [MulZeroOneClass α] : MulZeroOneClass (ULift α) :=
                                                      /-
                                                        α : Type u
                                                        inst✝ : MulZeroOneClass α
                                                        ⊢ ∀ (a b : ULift.{?u.15, u} α), Eq (Equiv.ulift (HMul.hMul a b)) (HMul.hMul (E …
                                                      -/
  Equiv.ulift.injective.mulZeroOneClass _ rfl rfl (by intros; rfl)
                                                              /-
                                                                🎉 no goals
                                                              -/


instance monoidWithZero [MonoidWithZero α] : MonoidWithZero (ULift α) :=
  Equiv.ulift.injective.monoidWithZero _ rfl rfl (fun _ _ => rfl) fun _ _ => rfl


instance commMonoidWithZero [CommMonoidWithZero α] : CommMonoidWithZero (ULift α) :=
  Equiv.ulift.injective.commMonoidWithZero _ rfl rfl (fun _ _ => rfl) fun _ _ => rfl


instance groupWithZero [GroupWithZero α] : GroupWithZero (ULift α) :=
  Equiv.ulift.injective.groupWithZero _ rfl rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


instance commGroupWithZero [CommGroupWithZero α] : CommGroupWithZero (ULift α) :=
  Equiv.ulift.injective.commGroupWithZero _ rfl rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


