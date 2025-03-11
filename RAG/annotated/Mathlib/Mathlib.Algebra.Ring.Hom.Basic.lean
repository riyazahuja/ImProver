/-- `f : α →+* β` has a trivial codomain iff its range is `{0}`. -/
theorem codomain_trivial_iff_range_eq_singleton_zero : (0 : β) = 1 ↔ Set.range f = {0} :=
  f.codomain_trivial_iff_range_trivial.trans
    ⟨fun h =>
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            x✝² : NonAssocSemiring α
                                            x✝¹ : NonAssocSemiring β
                                            f : RingHom α β
                                            h : ∀ (x : α), Eq (f x) 0
                                            y : β
                                            x✝ : Membership.mem (Set.range ⇑f) y
                                            x : α
                                            hx : Eq (f x) y
                                            ⊢ Membership.mem (Singleton.singleton 0) y
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
      Set.ext fun y => ⟨fun ⟨x, hx⟩ => by simp [← hx, h x], fun hy => ⟨0, by simpa using hy.symm⟩⟩,
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
      fun h x => Set.mem_singleton_iff.mp (h ▸ Set.mem_range_self x)⟩


protected theorem map_dvd (f : α →+* β) {a b : α} : a ∣ b → f a ∣ f b :=
  map_dvd f


/-- Pullback `IsDomain` instance along an injective function. -/
protected theorem Function.Injective.isDomain [Semiring α] [IsDomain α] [Semiring β] {F}
    [FunLike F β α] [MonoidWithZeroHomClass F β α] (f : F) (hf : Injective f) : IsDomain β where
  __ := domain_nontrivial f (map_zero _) (map_one _)
  __ := hf.isCancelMulZero f (map_zero _) (map_mul _)

