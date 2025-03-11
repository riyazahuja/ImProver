/-- If `f` is monotone on `(a,b]` and antitone on `[b,c)` then `f` has
a local maximum at `b`. -/
lemma isLocalMax_of_mono_anti
    {α : Type*} [TopologicalSpace α] [LinearOrder α] [OrderClosedTopology α]
    {β : Type*} [Preorder β]
    {a b c : α} (g₀ : a < b) (g₁ : b < c) {f : α → β}
    (h₀ : MonotoneOn f (Ioc a b))
    (h₁ : AntitoneOn f (Ico b c)) : IsLocalMax f b :=
  isLocalMax_of_mono_anti' (Ioc_mem_nhdsLE g₀) (Ico_mem_nhdsGE g₁) h₀ h₁


/-- If `f` is antitone on `(a,b]` and monotone on `[b,c)` then `f` has
a local minimum at `b`. -/
lemma isLocalMin_of_anti_mono
    {α : Type*} [TopologicalSpace α] [LinearOrder α] [OrderClosedTopology α]
    {β : Type*} [Preorder β] {a b c : α} (g₀ : a < b) (g₁ : b < c) {f : α → β}
    (h₀ : AntitoneOn f (Ioc a b)) (h₁ : MonotoneOn f (Ico b c)) : IsLocalMin f b :=
                                                       /-
                                                         α : Type u_1
                                                         inst✝³ : TopologicalSpace α
                                                         inst✝² : LinearOrder α
                                                         inst✝¹ : OrderClosedTopology α
                                                         β : Type u_2
                                                         inst✝ : Preorder β
                                                         a b c : α
                                                         g₀ : LT.lt a b
                                                         g₁ : LT.lt b c
                                                         f : α → β
                                                         h₀ : AntitoneOn f (Set.Ioc a b)
                                                         h₁ : MonotoneOn f (Set.Ico b c)
                                                         x : α
                                                         hx : Membership.mem (Set.Ioo a c) x
                                                         ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f b) (f x)) x) x
                                                       -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  mem_of_superset (Ioo_mem_nhds g₀ g₁) (fun x hx => by rcases le_total x b <;> aesop)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

