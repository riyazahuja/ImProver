@[deprecated "No deprecation message was provided." (since := "2024-09-11")]
instance [IsLeftCancel α₁ f] : IsLeftCancel β₁ (e.arrowCongr (e.arrowCongr e) f) :=
                                          /-
                                            α₁ : Type u_1
                                            β₁ : Type u_2
                                            e : Equiv α₁ β₁
                                            f : α₁ → α₁ → α₁
                                            inst✝ : IsLeftCancel α₁ f
                                            x y z : α₁
                                            ⊢ Eq ((e.arrowCongr (e.arrowCongr e)) f (e x) (e y)) ((e.arrowCongr (e.arrowCo …
                                          -/
  ⟨e.surjective.forall₃.2 fun x y z => by simpa using @IsLeftCancel.left_cancel _ f _ x y z⟩
                                          /-
                                            🎉 no goals
                                          -/


@[deprecated "No deprecation message was provided." (since := "2024-09-11")]
instance [IsRightCancel α₁ f] : IsRightCancel β₁ (e.arrowCongr (e.arrowCongr e) f) :=
                                          /-
                                            α₁ : Type u_1
                                            β₁ : Type u_2
                                            e : Equiv α₁ β₁
                                            f : α₁ → α₁ → α₁
                                            inst✝ : IsRightCancel α₁ f
                                            x y z : α₁
                                            ⊢ Eq ((e.arrowCongr (e.arrowCongr e)) f (e x) (e y)) ((e.arrowCongr (e.arrowCo …
                                          -/
  ⟨e.surjective.forall₃.2 fun x y z => by simpa using @IsRightCancel.right_cancel _ f _ x y z⟩
                                          /-
                                            🎉 no goals
                                          -/

