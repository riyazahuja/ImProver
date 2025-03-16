/-- The root expression of the position specified by the `GoalsLocation`. -/
def rootExpr : GoalsLocation → MetaM Expr
  | ⟨_, .hyp fvarId⟩        => fvarId.getType
  | ⟨_, .hypType fvarId _⟩  => fvarId.getType
  | ⟨_, .hypValue fvarId _⟩ => do return (← fvarId.getDecl).value
  | ⟨mvarId, .target _⟩     => mvarId.getType


/-- The `SubExpr.Pos` specified by the `GoalsLocation`. -/
def pos : GoalsLocation → Pos
  | ⟨_, .hyp _⟩          => .root
  | ⟨_, .hypType _ pos⟩  => pos
  | ⟨_, .hypValue _ pos⟩ => pos
  | ⟨_, .target pos⟩     => pos


/-- The hypothesis specified by the `GoalsLocation`. -/
def location : GoalsLocation → MetaM (Option Name)
  | ⟨_, .hyp fvarId⟩        => some <$> fvarId.getUserName
  | ⟨_, .hypType fvarId _⟩  => some <$> fvarId.getUserName
  | ⟨_, .hypValue fvarId _⟩ => some <$> fvarId.getUserName
  | ⟨_, .target _⟩          => return none


